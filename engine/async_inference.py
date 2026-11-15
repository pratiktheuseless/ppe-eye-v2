"""Multi-threaded inference: readers -> IQueue -> inference worker -> OQueue."""
import queue
import threading
import time
import cv2
from config import settings
from engine.compliance_logic import evaluate, compliance_rate
from engine.stream_manager import StreamReader

COLORS = {"SAFE": settings.GREEN, "ATTENTION": settings.YELLOW, "DANGER": settings.RED}


def annotate(frame, workers):
    for w in workers:
        x1, y1, x2, y2 = map(int, w.box)
        c = COLORS[w.status]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = w.status if not w.violations else f"{w.status}: {','.join(w.violations)}"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return frame


class AsyncInferenceEngine:
    def __init__(self, detector, sources: dict, on_incident=None, zones: dict | None = None):
        self.detector = detector
        self.iq = queue.Queue(settings.STREAM_QUEUE_SIZE)   # IQueue: raw frames
        self.oq = queue.Queue(settings.STREAM_QUEUE_SIZE)   # OQueue: annotated frames
        self.readers = [StreamReader(cid, src, self.iq) for cid, src in sources.items()]
        self.on_incident = on_incident
        self.zones = zones or {}
        self.stop_evt = threading.Event()
        self.fps = 0.0
        self._last_alert = {}   # cam -> timestamp (debounce)

    def start(self):
        for r in self.readers: r.start()
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def _infer_loop(self):
        n, t0 = 0, time.time()
        while not self.stop_evt.is_set():
            try:
                cam, frame = self.iq.get(timeout=0.5)
            except queue.Empty:
                continue
            workers = evaluate(self.detector.predict(frame))
            out = annotate(frame, workers)
            rate = compliance_rate(workers)
            bad = [w for w in workers if w.status != "SAFE"]
            if bad and self.on_incident and time.time() - self._last_alert.get(cam, 0) > 10:
                self._last_alert[cam] = time.time()
                worst = "DANGER" if any(w.status == "DANGER" for w in bad) else "ATTENTION"
                self.on_incident(cam, self.zones.get(cam, "Zone-?"),
                                 sorted({v for w in bad for v in w.violations}), worst, out.copy())
            try:
                self.oq.put_nowait((cam, out, rate))
            except queue.Full:
                try: self.oq.get_nowait()
                except queue.Empty: pass
                self.oq.put_nowait((cam, out, rate))
            n += 1
            if n % 30 == 0:
                self.fps = 30 / (time.time() - t0); t0 = time.time()

    def stop(self):
        self.stop_evt.set()
        for r in self.readers: r.stop()
