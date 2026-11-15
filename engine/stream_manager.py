import queue
import threading
import time
import cv2


class StreamReader(threading.Thread):
    """Reads a video file, webcam index or RTSP URL into a bounded queue (IQueue)."""

    def __init__(self, cam_id, source, iqueue: queue.Queue, loop=True):
        super().__init__(daemon=True)
        self.cam_id, self.source, self.q, self.loop = cam_id, source, iqueue, loop
        self.stop_evt = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.loop and not str(self.source).startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                if str(self.source).startswith("rtsp"):
                    time.sleep(1); cap.release(); cap = cv2.VideoCapture(self.source); continue
                break
            try:
                self.q.put((self.cam_id, frame), timeout=0.05)
            except queue.Full:
                try: self.q.get_nowait()   # drop oldest -> keep latency low
                except queue.Empty: pass
        cap.release()

    def stop(self):
        self.stop_evt.set()
