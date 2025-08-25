import torch, numpy as np, cv2
from ultralytics import YOLO
from unidepth.models import UniDepthV2
#just a test
# Load models
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
yolo_model = YOLO("yolo_models/yolo11n-uav-vehicle-bbox.pt")  # fine-tuned

# Jetson-friendly settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
model.resolution_level = 5                   
torch.backends.cudnn.benchmark = True 

# Load video
video_path = "videos/Waterloo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output at 1 FPS
output_path = "output/output_waterloo_jetson.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 1.0, (width, height))

# Process ~1 frame per second
step = max(1, int(round(fps)))
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % step != 0:   # skip frames
        frame_idx += 1
        continue

    # YOLO detection (smaller imgsz for speed)
    yres = yolo_model(frame, conf=0.25, imgsz=640, verbose=False)[0]
    boxes = yres.boxes
    if boxes is None or len(boxes) == 0:
        out.write(frame); frame_idx += 1; continue

    # Depth Estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model.infer(rgb_tensor, camera=None)  # No K
        depth_map = pred["depth"].squeeze().cpu().numpy()

    H, W = depth_map.shape
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        label = yolo_model.names[cls_id] if hasattr(yolo_model, "names") else "veh"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        depth_crop = depth_map[y1:y2, x1:x2]
        valid = depth_crop[np.isfinite(depth_crop) & (depth_crop > 0)]
        label_text = f"{label} {conf:.2f}, {np.median(valid):.1f}m" if valid.size else f"{label} {conf:.2f}, n/a"
        cv2.putText(frame, label_text, (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)
    frame_idx += 1

cap.release(); out.release(); cv2.destroyAllWindows()