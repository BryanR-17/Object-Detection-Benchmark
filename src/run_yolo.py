from ultralytics import YOLO

# load pretrained YOLOv8 model
model = YOLO("models/yolov8n.pt")

# run inference on VOC test images
results = model.predict(
    source="datasets/images/test2007",
    imgsz=640,
    save=True,
    conf=0.25
)

print("Inference complete!")