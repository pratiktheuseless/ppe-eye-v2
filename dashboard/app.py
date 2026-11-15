"""Run: streamlit run dashboard/app.py"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import streamlit as st
from config import settings
from dashboard import database as db
from dashboard.components import widgets
from engine.async_inference import AsyncInferenceEngine

st.set_page_config("PPE-EYE v2", layout="wide")
db.init_db()
st.title("PPE-EYE v2 - Construction Site Safety Monitor")

with st.sidebar:
    model = st.selectbox("Model", ["yolo11x", "faster_rcnn", "ssd"])
    weights = st.text_input("Weights path", str(settings.SAVED_MODELS / "best.pt"))
    sources = st.text_area("Camera sources (name=path/RTSP, one per line)", "Cam-1=sample.mp4")
    run = st.toggle("Run live monitoring")


@st.cache_resource
def get_engine(model, weights, sources):
    from models.train_benchmark import REGISTRY
    det = REGISTRY[model]().load(weights if Path(weights).exists() else None)
    srcs = dict(l.split("=", 1) for l in sources.splitlines() if "=" in l)
    zones = {c: f"Zone-{i + 1}" for i, c in enumerate(srcs)}

    def on_incident(cam, zone, viol, sev, frame):
        snap = settings.ROOT / "dashboard" / f"snap_{cam}_{int(time.time())}.jpg"
        cv2.imwrite(str(snap), frame)
        db.log_incident(cam, zone, viol, sev, str(snap))
    eng = AsyncInferenceEngine(det, srcs, on_incident, zones); eng.start()
    return eng


tab1, tab2, tab3, tab4 = st.tabs(["Live", "Incidents", "Analytics", "Active learning"])
with tab1:
    if run:
        eng = get_engine(model, weights, sources)
        slot, mslot, frames, rates = st.empty(), st.empty(), {}, {}
        for _ in range(300):   # refresh loop; Streamlit reruns on interaction
            try:
                cam, img, rate = eng.oq.get(timeout=1)
            except Exception:
                continue
            frames[cam] = img; rates[cam] = rate
            db.log_metric(cam, eng.zones.get(cam, "?"), rate)
            with slot.container(): widgets.live_grid(frames)
            with mslot.container():
                widgets.metrics_row(sum(rates.values()) / len(rates), eng.fps,
                                    sum(i["status"] == "Active" for i in db.incidents(100)))
    else:
        st.info("Enable 'Run live monitoring' in the sidebar.")
with tab2: widgets.incident_table()
with tab3: widgets.analytics()
with tab4: widgets.feedback_panel()
