import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from config import settings
from .base_detector import BaseDetector, Detection


class SSDMobileNetDetector(BaseDetector):
    """Lightweight edge detector.

    torchvision ships SSDLite with a MobileNet backbone (V3-Large); it is used
    as the closest built-in stand-in for SSD MobileNet V2. For a strict
    MobileNetV2-SSD, export a TF Object Detection API model to TFLite.
    `quantize_int8()` emulates the INT8 edge pipeline on CPU.
    """
    name = "ssd_mobilenet_v2"

    def __init__(self, class_map=None):
        self.class_map = class_map or settings.CHVG
        self.model = None

    def load(self, weights=None):
        m = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone="DEFAULT",
                                          num_classes=len(self.class_map) + 1)
        if weights:
            m.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model = m.to(settings.DEVICE).eval()
        return self

    @torch.no_grad()
    def predict(self, frame):
        t = torch.from_numpy(frame[..., ::-1].copy()).permute(2, 0, 1).float() / 255
        out = self.model([t.to(settings.DEVICE)])[0]
        return [Detection(self.class_map[int(l) - 1], float(s), tuple(b.tolist()))
                for b, s, l in zip(out["boxes"], out["scores"], out["labels"])
                if s >= settings.CONF_THRES]

    def train(self, loader, epochs=30, lr=0.01, **kw):
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
        for ep in range(epochs):
            for imgs, targets in loader:
                imgs = [i.to(settings.DEVICE) for i in imgs]
                targets = [{k: v.to(settings.DEVICE) for k, v in t.items()} for t in targets]
                loss = sum(self.model(imgs, targets).values())
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"epoch {ep + 1}/{epochs} loss={loss.item():.4f}")
        self.model.eval()

    def quantize_int8(self):
        self.model = torch.ao.quantization.quantize_dynamic(
            self.model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
        return self
