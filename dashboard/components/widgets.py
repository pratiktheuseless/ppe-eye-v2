import cv2
import pandas as pd
import streamlit as st
from dashboard import database as db


def live_grid(frames: dict, cols=2):
    cs = st.columns(cols)
    for i, (cam, img) in enumerate(frames.items()):
        cs[i % cols].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=cam, use_container_width=True)


def metrics_row(rate, fps, n_active):
    a, b, c = st.columns(3)
    a.metric("Compliance", f"{rate:.0f}%"); b.metric("FPS", f"{fps:.1f}"); c.metric("Active incidents", n_active)


def incident_table():
    rows = db.incidents(50)
    if not rows:
        st.info("No incidents logged."); return
    st.dataframe(pd.DataFrame(rows).drop(columns=["snapshot"]), use_container_width=True)
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    iid = c1.number_input("Incident ID", min_value=1, step=1)
    if c2.button("Resolve"): db.set_status(int(iid), "Resolved"); st.rerun()
    if c3.button("Escalate"): db.set_status(int(iid), "Escalated"); st.rerun()
    if c4.button("Reopen"): db.set_status(int(iid), "Active"); st.rerun()


def analytics():
    c1, c2, c3 = st.columns(3)
    t = db.trend_7d()
    c1.subheader("7-day compliance")
    if t: c1.line_chart(pd.DataFrame(t).set_index("day"))
    v = db.violation_breakdown()
    c2.subheader("Violation types")
    if v: c2.bar_chart(pd.Series(v))
    z = db.zone_compare()
    c3.subheader("Zone comparison")
    if z: c3.bar_chart(pd.DataFrame(z).set_index("zone"))


def feedback_panel():
    st.write("Flag misclassified frames; they are staged for the next 24-hour retraining cycle.")
    up = st.file_uploader("Upload frame", type=["jpg", "png"])
    note = st.text_input("What was wrong? (e.g. helmet missed)")
    if up and st.button("Stage for retraining"):
        from config import settings
        settings.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        p = settings.FEEDBACK_DIR / up.name
        p.write_bytes(up.getvalue()); db.add_feedback(str(p), note)
        st.success("Staged.")
    st.caption(f"{len(db.pending_feedback())} frames awaiting retraining")
