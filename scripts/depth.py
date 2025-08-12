import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from unidepth.models import UniDepthV2
from unidepth.utils import colorize
import cv2

# Load models
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
yolo_model = YOLO("yolo_models/yolov8n.pt")

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load video
video_path = "videos/Drone.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare output
output_path = "output/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection
    results = yolo_model(frame, conf=0.2, classes=[2, 3, 5, 7])[0] #can change confidence and classes?
    boxes = results.boxes

    # Convert to RGB for depth model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    rgb_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(device)

    # Depth Estimation
    with torch.no_grad():
        predictions = model.infer(rgb_tensor)
        depth_map = predictions["depth"].squeeze().cpu().numpy()

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        label = yolo_model.names[cls_id]

        if label in ["car", "truck", "bus", "motorbike"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract depth inside the box
            depth_crop = depth_map[y1:y2, x1:x2]
            valid_depths = depth_crop[depth_crop > 1] # Bigger than 1m, can change?

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