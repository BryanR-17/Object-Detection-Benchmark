import time
import os
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

image_dir = "datasets/images/test2007"

image_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".jpg")
]

# use only first 300 images for benchmarking
image_paths = image_paths[:300]

start = time.time()

model.predict(source=image_paths, imgsz=640, conf=0.25, save=False, verbose=False)

end = time.time()

total_time = end - start
num_images = len(image_paths)

print(f"Images processed: {num_images}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per image: {total_time/num_images:.4f} seconds")
print(f"FPS: {num_images/total_time:.2f}")