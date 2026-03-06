# Object Detection Benchmark

This project benchmarks three modern object detection architectures and analyzes the trade-offs between detection accuracy and inference speed.

The models evaluated in this project are:

- YOLOv8 (single-stage detector)
- Faster R-CNN (two-stage detector)
- DETR (transformer-based detector)

The goal of this project is to evaluate how different object detection architectures perform under different hyperparameter settings and input resolutions.

---

# Research Questions

This project aims to answer the following questions:

1. Which object detection architecture performs best in terms of detection accuracy?
2. What is the trade-off between inference speed and detection accuracy across models?
3. How sensitive are detection models to key hyperparameters such as confidence threshold and Non-Maximum Suppression (NMS) threshold?
4. How does image resolution affect detection performance?

---

# Dataset

The models will be evaluated using the **Pascal VOC dataset**, a standard benchmark dataset for object detection.

The dataset contains:

- 20 object classes
- thousands of labeled images
- bounding box annotations for each object

Pascal VOC is widely used in object detection research and provides a balanced dataset for evaluating model performance.

---

# Models

## YOLOv8
YOLO (You Only Look Once) is a single-stage object detection model that performs detection in one pass through the network, making it extremely fast.

## Faster R-CNN
Faster R-CNN is a two-stage object detector that first proposes candidate regions and then classifies them. This approach typically produces higher accuracy but is slower than single-stage detectors.

## DETR
DETR (Detection Transformer) is a transformer-based object detection architecture that eliminates the need for many traditional components such as anchor boxes and non-maximum suppression.

---

# Experiments

Several experiments will be conducted to evaluate model performance.

### Model Comparison
Each model will be evaluated using the same dataset and evaluation metrics to compare overall performance.

### Confidence Threshold Sweep
Different confidence thresholds will be tested to observe how precision and recall change.

### NMS Threshold Sweep
The Non-Maximum Suppression threshold will be varied to evaluate its effect on detection quality.

### Resolution Analysis
Models will be evaluated using different image resolutions to analyze the trade-off between accuracy and inference speed.

---

# Evaluation Metrics

Model performance will be evaluated using the following metrics:

- Mean Average Precision (mAP)
- Precision
- Recall
- Inference Speed (Frames Per Second)

---

# Repository Structure
