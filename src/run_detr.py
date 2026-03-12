import os
import time
import torch
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection

# settings
IMAGE_DIR = "datasets/images/test2007"
OUTPUT_DIR = "results/sample_predictions/detr"
NUM_IMAGES = 300
CONF_THRESHOLD = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load pretrained DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.to(device)
model.eval()

# collect images
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
])[:NUM_IMAGES]

total_time = 0.0

for i, filename in enumerate(image_files):

    image_path = os.path.join(IMAGE_DIR, filename)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()

    total_time += (end - start)

    # convert predictions
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=CONF_THRESHOLD
    )[0]

    # save sample predictions
    if i < 8:
        draw = ImageDraw.Draw(image)

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            text = f"{model.config.id2label[label.item()]} {score:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            draw.text((x1, y1), text, fill="blue")

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