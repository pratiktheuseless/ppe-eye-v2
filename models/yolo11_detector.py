from ultralytics import YOLO
from config import settings
from .base_detector import BaseDetector, Detection


class YOLO11Detector(BaseDetector):
    name = "yolo11x"

    def __init__(self, class_map=None):
        self.class_map = class_map or settings.CHVG
        self.model = None

    def load(self, weights=None):
        self.model = YOLO(weights or "yolo11x.pt")
        return self

    def predict(self, frame):
        r = self.model.predict(frame, conf=settings.CONF_THRES, iou=settings.IOU_THRES,
                               device=settings.DEVICE, verbose=False)[0]
        names = r.names
        return [Detection(names[int(c)], float(s), tuple(map(float, b)))
                for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy())]

    def train(self, data_yaml, epochs=100, **kw):
        # Mosaic, HSV jitter and 30 deg rotation via the Ultralytics trainer
        return self.model.train(data=data_yaml, epochs=epochs, imgsz=settings.IMG_SIZE,
                                mosaic=1.0, degrees=30, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                                device=settings.DEVICE, **kw)
