import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import settings
from .base_detector import BaseDetector, Detection


class FasterRCNNDetector(BaseDetector):
    """Two-stage detector: ResNet50-FPN backbone + RPN + region classifier."""
    name = "faster_rcnn_r50"

    def __init__(self, class_map=None):
        self.class_map = class_map or settings.CHVG
        self.num_classes = len(self.class_map) + 1  # +1 background
        self.model = None

    def load(self, weights=None):
        m = fasterrcnn_resnet50_fpn(weights="DEFAULT" if weights is None else None)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, self.num_classes)
        if weights:
            m.load_state_dict(torch.load(weights, map_location=settings.DEVICE))
        self.model = m.to(settings.DEVICE).eval()
        return self

    @torch.no_grad()
    def predict(self, frame):
        t = torch.from_numpy(frame[..., ::-1].copy()).permute(2, 0, 1).float() / 255
        out = self.model([t.to(settings.DEVICE)])[0]
        return [Detection(self.class_map[int(l) - 1], float(s), tuple(b.tolist()))
                for b, s, l in zip(out["boxes"], out["scores"], out["labels"])
                if s >= settings.CONF_THRES]

    def train(self, loader, epochs=20, lr=0.005, **kw):
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        for ep in range(epochs):
            for imgs, targets in loader:
                imgs = [i.to(settings.DEVICE) for i in imgs]
                targets = [{k: v.to(settings.DEVICE) for k, v in t.items()} for t in targets]
                loss = sum(self.model(imgs, targets).values())
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"epoch {ep + 1}/{epochs} loss={loss.item():.4f}")
        self.model.eval()
