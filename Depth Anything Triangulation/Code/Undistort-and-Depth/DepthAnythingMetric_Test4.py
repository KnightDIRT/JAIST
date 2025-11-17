import sys
import cv2
import torch
import time
import numpy as np
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", message="xFormers is not available")

# --------------------------------------------------
# Model setup
# --------------------------------------------------
repo_path = r"C:/Users/Torenia/Depth-Anything/metric_depth"
sys.path.insert(0, repo_path)

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

pretrained_resource = "local::C:/Users/Torenia/OneDrive/Documents/GitHub/JAIST/Depth Anything Triangulation/Code/models/depth_anything_metric_depth_indoor.pt"
config = get_config("zoedepth", "eval", "nyu", pretrained_resource=pretrained_resource)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(config).to(device).eval().half()
torch.backends.cudnn.benchmark = True

# --------------------------------------------------
# Colorbar
# --------------------------------------------------
def create_colorbar(height=480, width=80, min_val=0.0, max_val=10.0):
    gradient = np.linspace(1, 0, height)[:, None]
    cmap = cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    colorbar = cv2.resize(cmap, (width, height))

    # Add distance labels
    for i, label in enumerate(np.linspace(min_val, max_val, 6)):
        y = int(height - (i / 5) * height)
        cv2.putText(colorbar, f"{label:.1f}m", (5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return colorbar

colorbar = create_colorbar(480, 80, 0, 10)

# --------------------------------------------------
# Camera setup
# --------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open camera")
    sys.exit()

cv2.namedWindow("ZoeDepth - RGB | Depth | Scale", cv2.WINDOW_NORMAL)
cursor_pos = (0, 0)

def on_mouse(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_pos = (x, y)

cv2.setMouseCallback("ZoeDepth - RGB | Depth | Scale", on_mouse)

# --------------------------------------------------
# GPU colormap LUT (magma)
# --------------------------------------------------
cmap_np = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_MAGMA)
cmap_torch = torch.from_numpy(cmap_np[:, 0:3]).to(device).float() / 255.0  # (256,3)

# --------------------------------------------------
# Main loop
# --------------------------------------------------
print("✅ GPU-accelerated ZoeDepth started — Press 'q' to quit.")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to tensor directly on GPU
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gpu = torch.from_numpy(frame_rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.half, non_blocking=True) / 255.0

    # Resize on GPU
    img_t = F.interpolate(frame_gpu, (480, 640), mode='bilinear', align_corners=False)

    # Run model
    with torch.no_grad():
        pred = model(img_t)
        if isinstance(pred, dict):
            pred = pred.get("metric_depth", pred.get("out"))

    # Resize depth on GPU
    depth = F.interpolate(pred, size=(480, 640), mode='bilinear', align_corners=False)

    # Normalize to 0–255 on GPU
    dmin, dmax = depth.min(), depth.max()
    depth_norm = ((depth - dmin) / (dmax - dmin + 1e-8) * 255.0).clamp(0, 255).to(torch.uint8)

    # Apply GPU colormap via lookup
    idx = depth_norm.squeeze(0).squeeze(0).long()  # indices 0–255
    depth_color = cmap_torch[idx]  # shape: (480, 640, 3)


    # Convert back to CPU for display
    depth_vis = (depth_color * 255).byte().cpu().numpy()
    depth_vis = np.squeeze(depth_vis)
    rgb_frame = (img_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    combined = np.hstack((rgb_frame, depth_vis, colorbar))

    fps = 1.0 / (time.time() - t0)
    cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show depth at cursor
    x, y = cursor_pos
    depth_value = None

    if 0 <= y < 480:
        if 0 <= x < 640:  # Cursor on left (RGB)
            dx = int((x / 640) * depth.shape[3])
            dy = int((y / 480) * depth.shape[2])
            depth_value = depth[0, 0, dy, dx].item()

        elif 640 <= x < 1280:  # Cursor on middle (Depth map)
            dx = int(((x - 640) / 640) * depth.shape[3])
            dy = int((y / 480) * depth.shape[2])
            depth_value = depth[0, 0, dy, dx].item()

    # Display numeric depth
    if depth_value is not None:
        # Normalize to 0–10 m to match colorbar
        dmin, dmax = depth.min(), depth.max()
        val = (depth_value - dmin.item()) / (dmax.item() - dmin.item() + 1e-8)
        depth_m = val * 10.0  # scaled meters
        cv2.putText(combined, f"Depth: {depth_m:.2f} m",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow("ZoeDepth - RGB | Depth | Scale", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
