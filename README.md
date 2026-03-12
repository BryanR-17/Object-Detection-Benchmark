# Object Detection Benchmark

This project compares different object detection architectures on the Pascal VOC dataset.

Models evaluated:
- YOLOv8 (single-stage detector)
- Faster R-CNN (two-stage detector)
- DETR (transformer-based detector)

The goal is to compare the trade-offs between detection accuracy and inference speed.

Dataset:
Pascal VOC 2007 + 2012

Evaluation metrics:
- inference speed (FPS)
- runtime per image
- qualitative detection results

## Repository Structure

datasets/ → Pascal VOC dataset  
models/ → pretrained model weights  
src/ → experiment scripts  
results/ → benchmark outputs and figures  

## Baseline Results

YOLOv8n baseline:

Images evaluated: 100  
Average inference time: 0.08 seconds  
FPS: 12.49

Hardware: laptop CPU