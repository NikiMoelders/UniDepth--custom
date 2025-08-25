import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from unidepth.models import UniDepthV2
from unidepth.utils import colorize
import cv2
from unidepth.utils.camera import Pinhole

#focal length of 925, need to check this, giving a depth of 0m roughly
# Load models
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
yolo_model = YOLO("yolo_models/yolo11n-uav-vehicle-bbox.pt") # This is fine tuned

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Load video
video_path = "videos/Waterloo.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Camera intrinsics
fx_px = 925
fy_px = 925
cx = width / 2
cy = height / 2
K = torch.tensor([[923.54, 0, 641.05],
                  [0, 928.46, 364.73],
                  [0, 0, 1]], dtype=torch.float32, device=device)
camera=K.float().to(device).unsqueeze(0)

# Prepare output
output_path = "output/output_waterloo_ft.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection
    results = yolo_model(frame, conf=0.2)[0]  # can change confidence
    boxes = results.boxes

    # Convert to RGB for depth model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_tensor = (torch.from_numpy(frame_rgb)
                  .permute(2, 0, 1).unsqueeze(0).float().to(device))

    # Depth Estimation 
    with torch.no_grad():
        predictions = model.infer(rgb_tensor)  # no K
        depth_map = predictions["depth"].squeeze().cpu().numpy()
        #print("K_pred:", predictions["intrinsics"][0].cpu().numpy()) # Debugging
        print(f"Depth median: {np.nanmedian(depth_map):.3f} m, max: {np.nanmax(depth_map):.3f} m") # Debugging

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        label = yolo_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract depth inside the box 
        depth_crop = depth_map[y1:y2, x1:x2]
        valid_depths = depth_crop[np.isfinite(depth_crop) & (depth_crop > 0)]

        if valid_depths.size > 0:
            median_depth = np.median(valid_depths)
            label_text = f"{label} {conf:.2f}, {median_depth:.1f}m"
        else:
            label_text = f"{label} {conf:.2f}, n/a"

        # Add label text
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()