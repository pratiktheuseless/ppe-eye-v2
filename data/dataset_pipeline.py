"""Dataset pipeline: VOC XML / COCO JSON -> YOLO TXT, class balancing, augmentation."""
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# ---------- annotation converters ----------
def _yolo_line(cid, x1, y1, x2, y2, w, h):
    return f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def voc_to_yolo(xml_path, class_names: list[str]) -> list[str]:
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text); h = float(root.find("size/height").text)
    lines = []
    for obj in root.iter("object"):
        name = obj.find("name").text
        if name not in class_names:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(_yolo_line(class_names.index(name), x1, y1, x2, y2, w, h))
    return lines


def coco_to_yolo(json_path, out_dir) -> dict:
    """Writes one .txt per image; returns {category_name: yolo_id}."""
    data = json.load(open(json_path))
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cid_map = {c["id"]: i for i, c in enumerate(cats)}
    imgs = {i["id"]: i for i in data["images"]}
    per_img: dict[int, list[str]] = {i: [] for i in imgs}
    for a in data["annotations"]:
        im = imgs[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        per_img[a["image_id"]].append(
            _yolo_line(cid_map[a["category_id"]], x, y, x + bw, y + bh, im["width"], im["height"]))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for iid, lines in per_img.items():
        (out_dir / (Path(imgs[iid]["file_name"]).stem + ".txt")).write_text("\n".join(lines))
    return {c["name"]: cid_map[c["id"]] for c in cats}


# ---------- augmentation ----------
def hsv_jitter(img, h=0.015, s=0.7, v=0.4):
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
    hsv[..., 1:] = np.clip(hsv[..., 1:] * r[1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def rotate_with_boxes(img, boxes, max_deg=30):
    """boxes: Nx4 xyxy pixels. Rotates image and re-fits axis-aligned boxes."""
    h, w = img.shape[:2]
    ang = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    new = []
    for x1, y1, x2, y2 in boxes:
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ M.T
        new.append([max(0, pts[:, 0].min()), max(0, pts[:, 1].min()),
                    min(w, pts[:, 0].max()), min(h, pts[:, 1].max())])
    return out, np.array(new)


def mosaic(imgs, size=640):
    """Combine 4 images into a 2x2 mosaic (labels must be re-mapped by caller)."""
    assert len(imgs) == 4
    half = size // 2
    tiles = [cv2.resize(i, (half, half)) for i in imgs]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


# ---------- class balancing ----------
def class_counts(label_dir) -> Counter:
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    return c


def oversample_minority(label_dir, target_ratio=0.5):
    """Object-detection analogue of SMOTE.

    Classic SMOTE interpolates feature vectors, which is not meaningful for raw
    images/boxes. We oversample images that contain under-represented classes
    (listed for duplication + augmentation) until each class reaches
    `target_ratio` of the majority class. Returns a list of label files to duplicate.
    """
    counts = class_counts(label_dir)
    if not counts:
        return []
    target = int(max(counts.values()) * target_ratio)
    by_class: dict[int, list[Path]] = {}
    for f in Path(label_dir).glob("*.txt"):
        for cid in {int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()}:
            by_class.setdefault(cid, []).append(f)
    dup = []
    for cid, n in counts.items():
        if n < target and by_class.get(cid):
            need = (target - n) // max(1, n // len(by_class[cid]))
            dup += [random.choice(by_class[cid]) for _ in range(min(need, 5 * len(by_class[cid])))]
    return dup


def smote_feature_balance(X, y):
    """True SMOTE on feature vectors (e.g. CNN embeddings / crop features) via imbalanced-learn."""
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=0).fit_resample(X, y)


def write_dataset_yaml(path, root, names: dict):
    lines = [f"path: {root}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {k}: {v}" for k, v in names.items()]
    Path(path).write_text("\n".join(lines) + "\n")
