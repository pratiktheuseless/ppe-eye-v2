import sqlite3
import threading
from datetime import datetime, timedelta
from config import settings

_lock = threading.Lock()


def _conn():
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS incidents(
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, camera TEXT, zone TEXT,
          violations TEXT, severity TEXT, status TEXT DEFAULT 'Active', snapshot TEXT);
        CREATE TABLE IF NOT EXISTS metrics(
          ts TEXT, camera TEXT, zone TEXT, compliance REAL);
        CREATE TABLE IF NOT EXISTS feedback(
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, image_path TEXT, note TEXT, used INTEGER DEFAULT 0);
        """)


def log_incident(camera, zone, violations, severity, snapshot=None):
    with _lock, _conn() as c:
        c.execute("INSERT INTO incidents(ts,camera,zone,violations,severity,snapshot) VALUES(?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), camera, zone,
                   ",".join(violations), severity, snapshot))


def set_status(incident_id, status):
    assert status in ("Active", "Resolved", "Escalated")
    with _lock, _conn() as c:
        c.execute("UPDATE incidents SET status=? WHERE id=?", (status, incident_id))


def incidents(limit=100):
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))]


def log_metric(camera, zone, compliance):
    with _lock, _conn() as c:
        c.execute("INSERT INTO metrics VALUES(?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), camera, zone, compliance))


def trend_7d():
    since = (datetime.now() - timedelta(days=7)).isoformat()
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT substr(ts,1,10) day, ROUND(AVG(compliance),1) compliance FROM metrics "
            "WHERE ts>=? GROUP BY day ORDER BY day", (since,))]


def zone_compare():
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT zone, ROUND(AVG(compliance),1) compliance FROM metrics GROUP BY zone")]


def violation_breakdown():
    out = {}
    for r in incidents(10000):
        for v in filter(None, r["violations"].split(",")):
            out[v] = out.get(v, 0) + 1
    return out


def add_feedback(image_path, note=""):
    with _lock, _conn() as c:
        c.execute("INSERT INTO feedback(ts,image_path,note) VALUES(?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), image_path, note))


def pending_feedback():
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM feedback WHERE used=0")]
