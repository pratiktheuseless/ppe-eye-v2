"""Central configuration for PPE-EYE v2."""
from pathlib import Path
import yaml

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # lets tests/logic run without torch
    DEVICE = "cpu"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAVED_MODELS = ROOT / "exports" / "saved_models"
DB_PATH = ROOT / "dashboard" / "ppe_eye.db"
FEEDBACK_DIR = ROOT / "data" / "feedback_queue"

CONF_THRES = 0.3
IOU_THRES = 0.5
IMG_SIZE = 640
STREAM_QUEUE_SIZE = 64

with open(ROOT / "config" / "classes_config.yaml") as f:
    _CLS = yaml.safe_load(f)
CHVG = _CLS["chvg"]
PICTOR = _CLS["pictor_v3"]
CCTV = _CLS["cctv"]

HELMET_CLASSES = {"white", "blue", "red", "yellow"}

# Compliance colours (BGR)
GREEN, YELLOW, RED = (0, 200, 0), (0, 215, 255), (0, 0, 230)

# Targets from the reference literature (for benchmark comparison tables)
TARGETS = {
    "yolo11x": {"mAP50": 0.969, "latency_ms": 7.3},
    "faster_rcnn_r50": {"mAP50": 0.96, "latency_ms": 170.0},
    "ssd_mobilenet_v2": {"mAP50": None, "latency_ms": 27.0},
}
