import pandas as pd
import matplotlib.pyplot as plt
import os

# paths
CSV_PATH = "results/tables/model_comparison.csv"
FIG_DIR = "results/figures"

os.makedirs(FIG_DIR, exist_ok=True)

# load data
df = pd.read_csv(CSV_PATH)

# -----------------------------
# FPS comparison plot
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["FPS"])
plt.title("Object Detection Model Speed (FPS)")
plt.ylabel("Frames Per Second")
plt.xlabel("Model")
plt.tight_layout()

fps_path = os.path.join(FIG_DIR, "fps_comparison.png")
plt.savefig(fps_path)
plt.close()

# -----------------------------
# inference time comparison
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Avg Time per Image (s)"])
plt.title("Average Inference Time per Image")
plt.ylabel("Seconds")
plt.xlabel("Model")
plt.tight_layout()

time_path = os.path.join(FIG_DIR, "inference_time_comparison.png")
plt.savefig(time_path)
plt.close()

print("Figures saved to:", FIG_DIR)

# -----------------------------
# total runtime comparison
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Total Runtime (s)"])
plt.title("Total Runtime for 100 Images")
plt.ylabel("Seconds")
plt.xlabel("Model")
plt.tight_layout()

runtime_path = os.path.join(FIG_DIR, "total_runtime_comparison.png")
plt.savefig(runtime_path)
plt.close()