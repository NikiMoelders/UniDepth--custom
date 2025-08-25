import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole  # if available

torch.backends.cudnn.benchmark = True

# Load models
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
yolo_model = YOLO("yolo_models/yolo11n-vsai-obb.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
yolo_model.to(device)

# Load video
video_path = "videos/Waterloo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Camera intrinsics (pixels)
fx_px = 925.0
fy_px = 925.0
cx = width / 2.0
cy = height / 2.0

# If your UniDepth build supports Pinhole, use it for metric depth:
try:
    pinhole = Pinhole(fx_px, fy_px, cx, cy, width, height).to(device).batch(1)
except Exception:
    pinhole = None  # falls back to up-to-scale depth

# Prepare output
os.makedirs("output", exist_ok=True)
output_path = "output/output_waterloo_obb.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO expects BGR np.ndarray; conf threshold as needed
    results = yolo_model(frame, conf=0.2, verbose=False, device=0 if device.type=="cuda" else "cpu")[0]

    boxes_xyxy, confs, cls_ids = [], [], []
    if getattr(results, "obb", None) is not None and len(results.obb) > 0:
        if getattr(results.obb, "xyxyxyxy", None) is not None:
            polys = results.obb.xyxyxyxy.cpu().numpy()
            confs = results.obb.conf.cpu().numpy().tolist()
            cls_ids = results.obb.cls.cpu().numpy().astype(int).tolist()
            for p in polys:
                xs, ys = p[0::2], p[1::2]
                boxes_xyxy.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        elif getattr(results.obb, "xywhr", None) is not None:
            xywhr = results.obb.xywhr.cpu().numpy()
            confs = results.obb.conf.cpu().numpy().tolist()
            cls_ids = results.obb.cls.cpu().numpy().astype(int).tolist()
            for cxr, cyr, w, h, r in xywhr:
                rect = ((float(cxr), float(cyr)), (float(w), float(h)), float(np.degrees(r)))
                pts = cv2.boxPoints(rect)  # (4,2)
                x1, y1 = float(pts[:,0].min()), float(pts[:,1].min())
                x2, y2 = float(pts[:,0].max()), float(pts[:,1].max())
                boxes_xyxy.append([x1, y1, x2, y2])
    elif results.boxes is not None and len(results.boxes) > 0:
        boxes_xyxy = results.boxes.xyxy.cpu().numpy().tolist()
        confs = results.boxes.conf.cpu().numpy().tolist()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int).tolist()

    if not boxes_xyxy:
        out.write(frame)
        continue

    # Depth Estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(frame_rgb).to(device).permute(2,0,1).unsqueeze(0).float() / 255.0
    rgb_tensor = rgb_tensor.to(memory_format=torch.channels_last)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
        # If Pinhole is available in your version, pass it; otherwise omit it.
        if pinhole is not None:
            predictions = model.infer(rgb_tensor, K=pinhole)
        else:
            predictions = model.infer(rgb_tensor)

        depth_map = predictions["depth"].squeeze()
        # Ensure HxW equals frame size
        depth_map = torch.nn.functional.interpolate(
            depth_map[None, None], size=(height, width), mode="nearest"
        ).squeeze().float().cpu().numpy()

    # Optional debug
    # print(f"Depth median: {np.nanmedian(depth_map):.3f} m, max: {np.nanmax(depth_map):.3f} m")

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(width - 1, x2)), int(min(height - 1, y2))

        label = yolo_model.model.names[cls_id] if hasattr(yolo_model, "model") and hasattr(yolo_model.model, "names") else (
                 yolo_model.names[cls_id] if hasattr(yolo_model, "names") else "obj"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        depth_crop = depth_map[y1:y2, x1:x2]
        valid = np.isfinite(depth_crop) & (depth_crop > 0)
        if valid.any():
            median_depth = float(np.median(depth_crop[valid]))
            label_text = f"{label} {float(conf):.2f}, {median_depth:.1f}m"
        else:
            label_text = f"{label} {float(conf):.2f}, n/a"

        cv2.putText(frame, label_text, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Wrote {output_path}")
