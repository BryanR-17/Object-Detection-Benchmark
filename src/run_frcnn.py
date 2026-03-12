import os
import time
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# -----------------------------
# settings
# -----------------------------
IMAGE_DIR = "datasets/images/test2007"
OUTPUT_DIR = "results/sample_predictions/frcnn"
NUM_IMAGES = 300
SCORE_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# model
# -----------------------------
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()

# COCO class names for pretrained Faster R-CNN
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# -----------------------------
# collect images
# -----------------------------
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
])[:NUM_IMAGES]

total_time = 0.0

for i, filename in enumerate(image_files):
    image_path = os.path.join(IMAGE_DIR, filename)

    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    start = time.time()
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    end = time.time()

    total_time += (end - start)

    # save a few sample predictions
    if i < 8:
        draw = ImageDraw.Draw(image)

        boxes = prediction["boxes"].cpu()
        scores = prediction["scores"].cpu()
        labels = prediction["labels"].cpu()

        for box, score, label in zip(boxes, scores, labels):
            if score < SCORE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.tolist()
            class_name = COCO_CLASSES[label.item()]
            text = f"{class_name}: {score:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), text, fill="red")

        save_path = os.path.join(OUTPUT_DIR, filename)
        image.save(save_path)

num_images = len(image_files)
avg_time = total_time / num_images
fps = num_images / total_time

print(f"Images processed: {num_images}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time:.4f} seconds")
print(f"FPS: {fps:.2f}")
print(f"Sample predictions saved to: {OUTPUT_DIR}")