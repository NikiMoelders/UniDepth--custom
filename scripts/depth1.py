import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from unidepth.models import UniDepthV2
from unidepth.utils import colorize

# ------------ Load models ----------------
depth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
yolo_model = YOLO("yolov8n.pt")

# ------------ Device setup ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model = depth_model.to(device).eval()

# ------------ Load image -----------------
image_path = "assets/demo/frame_006.png"
image = Image.open(image_path).convert("RGB")
rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)  # C, H, W

# ------------ YOLOv8 Inference -----------
results = yolo_model(image_path)[0]
boxes = results.boxes
print(f"Detected {len(boxes)} objects")

# ------------ Depth Estimation -----------
with torch.no_grad():
    predictions = depth_model.infer(rgb)
    depth_map = predictions["depth"].squeeze().cpu().numpy()  # H, W

# ------------ Annotate -------------------
draw = ImageDraw.Draw(image)

for i, box in enumerate(boxes):
    cls_id = int(box.cls[0].item())
    label = yolo_model.names[cls_id]

    if label != "car":
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)

    depth_crop = depth_map[y1:y2, x1:x2]
    valid_depths = depth_crop[depth_crop > 0.1]

    if valid_depths.size == 0:
        continue

    median_depth = np.median(valid_depths)
    print(f"[{label}] at box {x1,y1,x2,y2} → {median_depth:.2f} meters")

    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.text((x1, y1 - 10), f"{label}: {median_depth:.1f} m", fill="yellow")

# ------------ Save Output ----------------
image.save("annotated_image.png")
print("Saved: annotated_image.png")
