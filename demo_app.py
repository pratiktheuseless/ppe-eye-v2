"""PPE-EYE v2 - upload demo.  Run:  streamlit run demo_app.py

Upload a photo or a short video, get the same green / yellow / red boxes,
a summary and a worker table. Put your trained model at exports/saved_models/best.pt
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from engine.compliance_logic import evaluate, compliance_rate
from engine.async_inference import annotate

st.set_page_config("PPE-EYE v2 Demo", layout="wide")
st.title("PPE-EYE v2 - Safety Check Demo")
st.caption("Upload a construction-site photo or video. Green = safe, yellow = attention, red = danger.")

with st.sidebar:
    weights = st.text_input("Model file", str(settings.SAVED_MODELS / "best.pt"))
    conf = st.slider("Detection confidence", 0.10, 0.80, 0.30, 0.05)
    strict = st.checkbox("Strict vest check (flag any worker without a detected vest)", value=False)
    skip = st.slider("Video: process every Nth frame", 1, 10, 3)
    max_sec = st.slider("Video: max seconds", 5, 60, 20)


@st.cache_resource
def load_detector(path):
    from models.yolo11_detector import YOLO11Detector
    return YOLO11Detector().load(path)


def analyse(det, frame):
    settings.CONF_THRES = conf
    workers = evaluate(det.predict(frame), strict=strict)
    return annotate(frame.copy(), workers), workers


def summary(workers_all):
    n = len(workers_all)
    safe = sum(w.status == "SAFE" for w in workers_all)
    att = sum(w.status == "ATTENTION" for w in workers_all)
    dang = sum(w.status == "DANGER" for w in workers_all)
    a, b, c, d = st.columns(4)
    a.metric("Workers seen", n); b.metric("Safe", safe)
    c.metric("Attention", att); d.metric("Danger", dang)
    st.progress(min(1.0, safe / n) if n else 1.0, text=f"Compliance: {compliance_rate(workers_all):.0f}%")


if not Path(weights).exists():
    st.error(f"Model file not found: {weights}. Copy best.pt into exports/saved_models/ or fix the path.")
    st.stop()
det = load_detector(weights)

up = st.file_uploader("Upload a photo or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
if up is None:
    st.info("Waiting for an upload.")
    st.stop()

if up.type.startswith("image"):
    img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    out, workers = analyse(det, img)
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    c2.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
    summary(workers)
    if workers:
        st.dataframe(pd.DataFrame([{"Worker": i + 1, "Status": w.status,
                                    "Problems": ", ".join(w.violations) or "none"}
                                   for i, w in enumerate(workers)]), use_container_width=True)
else:
    tmp = Path(tempfile.mkdtemp())
    src = tmp / ("in" + Path(up.name).suffix)
    src.write_bytes(up.getvalue())
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    raw = tmp / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps / skip, (w, h))
    limit = int(max_sec * fps)
    bar = st.progress(0.0, text="Processing video...")
    all_workers, keyframes, n = [], [], 0
    while n < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if n % skip == 0:
            out, workers = analyse(det, frame)
            writer.write(out)
            all_workers += workers
            if len(keyframes) < 4 and workers and n % (skip * 15) == 0:
                keyframes.append(out)
        n += 1
        bar.progress(min(1.0, n / limit), text=f"Processing video... frame {n}")
    cap.release(); writer.release(); bar.empty()

    web = tmp / "web.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:   # bundled ffmpeg from the imageio-ffmpeg package (pip install imageio-ffmpeg)
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-i", str(raw), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(web)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    final = web if web.exists() and web.stat().st_size > 0 else raw
    if final == web:
        st.video(str(final))
    else:
        st.warning("Browser preview needs ffmpeg. Showing key frames; the video can still be downloaded.")
        for k in keyframes:
            st.image(cv2.cvtColor(k, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.download_button("Download result video", final.read_bytes(), "ppe_result.mp4", "video/mp4")
    summary(all_workers)
    if all_workers:
        counts = pd.Series([v for x in all_workers for v in x.violations]).value_counts()
        if len(counts):
            st.subheader("Violation types (worker-detections)")
            st.bar_chart(counts)
