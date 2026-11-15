"""PyTorch -> ONNX -> TensorRT.   python -m exports.export_onnx_trt --weights best.pt"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--half", action="store_true", help="FP16 TensorRT engine")
a = ap.parse_args()
m = YOLO(a.weights)
m.export(format="onnx", opset=17, simplify=True)
m.export(format="engine", half=a.half, device=0)   # requires NVIDIA GPU + TensorRT
