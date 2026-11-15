"""Unified train / benchmark entry point.

  python -m models.train_benchmark --model yolo11x --data data/processed/chvg.yaml --epochs 100
  python -m models.train_benchmark --benchmark --image sample.jpg
"""
import argparse
import json
import cv2
from config import settings
from models.yolo11_detector import YOLO11Detector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.ssd_mobilenet_detector import SSDMobileNetDetector

REGISTRY = {"yolo11x": YOLO11Detector, "faster_rcnn": FasterRCNNDetector, "ssd": SSDMobileNetDetector}


def benchmark(image_path):
    frame = cv2.imread(image_path)
    rows = []
    for cls in REGISTRY.values():
        det = cls().load()
        rows.append({"model": det.name, "latency_ms": round(det.benchmark_latency(frame), 2),
                     "target": settings.TARGETS[det.name]})
    print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=REGISTRY, default="yolo11x")
    ap.add_argument("--data", default="data/processed/chvg.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--image")
    a = ap.parse_args()
    if a.benchmark:
        benchmark(a.image)
    elif a.model == "yolo11x":
        YOLO11Detector().load().train(a.data, a.epochs)
    else:
        print("Faster R-CNN / SSD: build a torch DataLoader from data/processed and call .train(loader)")
