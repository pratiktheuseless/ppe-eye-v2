# PPE-EYE v2
AI and Computer Vision-Based Automated PPE and Construction Site Safety Monitoring System.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests            # compliance logic + converter tests
streamlit run dashboard/app.py    # dashboard
```

## Workflow
1. Put datasets (CHVG, Pictor-v3, CCTV helmet/vest sets) in `data/raw/`.
2. Convert with `data/dataset_pipeline.py` (`voc_to_yolo`, `coco_to_yolo`), balance with `oversample_minority`, write `data/processed/chvg.yaml` with `write_dataset_yaml`.
3. Train: `python -m models.train_benchmark --model yolo11x --data data/processed/chvg.yaml --epochs 100`
4. Benchmark: `python -m tests.benchmark_eval sample.jpg`
5. Export for edge: `python -m exports.export_onnx_trt --weights exports/saved_models/best.pt --half`
6. Put the trained weights at `exports/saved_models/best.pt` and run the dashboard.

## Upload demo (best for presentations)
```bash
streamlit run demo_app.py
```
Upload a photo or video; the page shows the coloured result, a summary and a worker table. Put your trained model at `exports/saved_models/best.pt` first.

## Architecture
cameras/video/RTSP -> `StreamReader` threads -> **IQueue** -> detector -> `compliance_logic` -> **OQueue** -> dashboard.
Incidents and metrics are stored in SQLite (`dashboard/database.py`).

## Compliance rules
Missing helmet = DANGER (red). Missing vest = ATTENTION (yellow). Eye protection is optional. Everything fine = SAFE (green).

## Honest notes (say these in your viva)
- mAP50 0.969 / 7.3 ms (YOLO11x), 96% / 0.17 s (Faster R-CNN), 27 ms (SSD) are the *literature targets* in `config/settings.py`. You must train on the real datasets and report your own numbers, which will depend on your GPU.
- "SMOTE" on raw images is not well defined. `oversample_minority` is the detection-friendly equivalent (duplicate and augment images with rare classes); `smote_feature_balance` is real SMOTE for feature vectors.
- SSD uses torchvision SSDLite (MobileNetV3 backbone) as a stand-in for SSD MobileNet V2. INT8 here is dynamic quantisation on CPU, not Ambarella hardware.
- TensorRT export needs an NVIDIA GPU.

## Presentation outline
Problem and statistics -> literature gap -> architecture diagram -> datasets and augmentation -> three-model comparison -> compliance logic -> dashboard demo -> results -> limitations and future work (pose-based checks, harness/fall detection, edge deployment).
