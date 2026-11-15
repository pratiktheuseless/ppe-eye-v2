# PPE-EYE v2

**AI and Computer Vision-Based Automated PPE and Construction Site Safety Monitoring System**

PPE-EYE v2 looks at a construction-site photo or video and checks whether each worker is wearing a helmet and a high-visibility vest. Every worker gets a coloured box:

| Colour | Status | Meaning |
|---|---|---|
| 🟢 Green | SAFE | Helmet and vest found |
| 🟡 Yellow | ATTENTION | Model is confident the worker has no vest |
| 🔴 Red | DANGER | No helmet found |

Built by Pratik Pathak as a college project.

## Demo

![Photo result](result-photo.png)
![Video frame](result-video.png)

## How it works

1. **Detect.** A YOLO11s object detector, fine-tuned on a construction-safety dataset, finds people, hardhats, vests and related items in each frame.
2. **Decide.** A small rule engine (`engine/compliance_logic.py`) matches helmets and vests to each person by box overlap and assigns SAFE, ATTENTION or DANGER.
3. **Show.** A Streamlit page (`demo_app.py`) lets you upload a photo or video and see the coloured result, a summary and a worker table.

## Results

Model: YOLO11s fine-tuned from COCO-pretrained weights on the Roboflow *Construction Site Safety Image Dataset* (10 classes, about 5,850 training photos). Trained for 20 epochs on a free Kaggle T4 GPU at 640 px. Scores are from the held-out test split.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | 0.767 | 0.598 | **0.647** | 0.358 |
| Hardhat | 0.955 | 0.763 | 0.875 | 0.508 |
| Safety Vest | 0.788 | 0.721 | 0.783 | 0.503 |
| Person | 0.744 | 0.684 | 0.710 | 0.425 |
| machinery | 0.778 | 0.568 | 0.692 | 0.471 |
| NO-Mask | 0.720 | 0.586 | 0.661 | 0.242 |
| NO-Safety Vest | 0.806 | 0.555 | 0.626 | 0.319 |
| Mask | 1.000 | 0.707 | 0.757 | 0.464 |
| NO-Hardhat | 0.706 | 0.561 | 0.485 | 0.205 |
| Safety Cone | 0.679 | 0.467 | 0.468 | 0.198 |
| vehicle | 0.492 | 0.366 | 0.409 | 0.240 |

mAP50 is a score from 0 to 1 (higher is better). These are student-project numbers on a small free-GPU training run, not production figures.

## Run it yourself

Requires Python 3.12. The trained model is included at `exports/saved_models/best.pt`.

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install ultralytics streamlit pyyaml pandas opencv-python imageio-ffmpeg
streamlit run demo_app.py
```

Open the page, upload a photo or a short video, and read the result. Photos work well on a normal laptop; videos run on CPU and are slower. The sidebar lets you change the detection confidence, skip frames and cap the video length.

Run the rule and converter tests with:

```bash
pip install pytest
python -m pytest tests
```

## Compliance rules

- **No helmet → DANGER.** A worker is flagged unless a helmet box overlaps the head region.
- **No vest → ATTENTION.** Raised only when the model is at least 60% sure of a "no vest" detection (`NO_VEST_CONF` in `engine/compliance_logic.py`). This avoids false alarms when the model is unsure, at the cost of missing some real vest violations. Tick *Strict vest check* in the sidebar to flag any worker without a detected vest instead.

## Project layout

```
config/       settings and class maps
data/         annotation converters (VOC/COCO to YOLO), augmentation, class balancing
models/       YOLO11, Faster R-CNN and SSDLite wrappers, train/benchmark script
engine/       threaded stream reader, async inference queues, compliance rules
dashboard/    Streamlit incident dashboard with SQLite logging
exports/      trained weights and ONNX/TensorRT export script
tests/        unit tests for the rules and converters
demo_app.py   upload demo
```

## Status

| Part | Status |
|---|---|
| Training, evaluation, trained model | Done |
| Upload demo (`demo_app.py`) | Working |
| Compliance rules and data converters | Unit-tested |
| Multi-camera engine, incident dashboard (`dashboard/app.py`) | Implemented, not fully tested |
| Faster R-CNN and SSDLite wrappers | Implemented, not trained or benchmarked |
| ONNX/TensorRT export | Implemented, needs an NVIDIA GPU, not run |

## Limitations

- Trained on one public dataset. On a test video with a different vest style and close-up framing, the model failed to detect a worn vest, which is why the vest rule needs a confidence threshold.
- When workers overlap, a vest can be matched to the wrong person.
- Some classes score low (vehicle, safety cone, NO-Hardhat). Recall is about 0.60 overall, so some objects are missed.
- Small model, short training run, free GPU. More data and a larger model would improve accuracy.

## Future work

Train on more varied data (more vest styles, camera angles, lighting), compare larger models and the Faster R-CNN / SSD baselines, add pose-based checks (harness, fall detection), and deploy on an edge device.

## Credits

Dataset: *Construction Site Safety Image Dataset* (Roboflow, via Kaggle). Check the dataset's licence before reusing it. Detector: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).
