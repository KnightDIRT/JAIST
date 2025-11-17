import sys
import cv2
import torch
import time
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="xFormers is not available")

# --------------------------------------------------
# 1. Model setup
# --------------------------------------------------
repo_path = r"C:/Users/Torenia/Depth-Anything/metric_depth"
sys.path.insert(0, repo_path)

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

pretrained_resource = (
    "local::C:/Users/Torenia/OneDrive/Documents/GitHub/JAIST/"
    "Depth Anything Triangulation/Code/models/depth_anything_metric_depth_indoor.pt"
)

torch.backends.cudnn.benchmark = True  # optimize CUDA performance

# Build model
config = get_config("zoedepth", "eval", "nyu", pretrained_resource=pretrained_resource)
model = build_model(config).cuda().eval()

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------
def create_colorbar(min_val=0.0, max_val=10.0, height=480, width=80):
    """Create a vertical colorbar showing distance values."""
    gradient = np.linspace(1, 0, height)[:, None]
    cmap = plt.get_cmap('magma')
    colorbar = (cmap(gradient)[:, :, :3] * 255).astype(np.uint8)
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_RGB2BGR)
    colorbar = cv2.resize(colorbar, (width, height))

    # Add distance labels
    for i, label in enumerate(np.linspace(min_val, max_val, 6)):
        y = int(height - (i / 5) * height)
        cv2.putText(colorbar, f"{label:.1f}m", (5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return colorbar

colorbar = create_colorbar(0, 10, 480, 80)

# --------------------------------------------------
# 3. Camera setup
# --------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    sys.exit()

print("✅ Real-time ZoeDepth started — Press 'q' to quit.")
fps = 0.0
cursor_pos = (0, 0)
depth_map = None

# --------------------------------------------------
# 4. Mouse callback
# --------------------------------------------------
def on_mouse(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_pos = (x, y)

cv2.namedWindow("ZoeDepth - RGB | Depth | Scale")
cv2.setMouseCallback("ZoeDepth - RGB | Depth | Scale", on_mouse)

# --------------------------------------------------
# 5. Main loop
# --------------------------------------------------
while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_pil = transforms.ToPILImage()(img_rgb)
    img_t = transform(img_pil).unsqueeze(0).cuda()

    # Run model
    with torch.no_grad():
        pred = model(img_t)
        if isinstance(pred, dict):
            pred = pred.get("metric_depth", pred.get("out"))
    depth_map = pred.squeeze().cpu().numpy()

    # Normalize depth for visualization
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    depth_vis = cv2.resize(depth_vis, (640, 480))

    # Combine images: RGB | Depth | Colorbar
    combined = np.hstack((frame_resized, depth_vis, colorbar))

    # Compute true FPS
    frame_end = time.time()
    fps = 1.0 / (frame_end - frame_start)

    # Draw FPS
    cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Depth under cursor (for either left or middle frame)
    x, y = cursor_pos
    depth_value = None
    if 0 <= y < 480:
        if 0 <= x < 640:  # Cursor on RGB
            # Map cursor to depth pixel coordinates
            dx = int((x / 640) * depth_map.shape[1])
            dy = int((y / 480) * depth_map.shape[0])
            depth_value = depth_map[dy, dx]

        elif 640 <= x < 1280:  # Cursor on Depth map
            # Map cursor from second panel (640-1280)
            dx = int(((x - 640) / 640) * depth_map.shape[1])
            dy = int((y / 480) * depth_map.shape[0])
            depth_value = depth_map[dy, dx]


    # Show depth text
    if depth_value is not None:
        cv2.putText(combined, f"Depth: {depth_value:.2f} m",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (200, 200, 200), 2, cv2.LINE_AA)

    # Show the combined result
    cv2.imshow("ZoeDepth - RGB | Depth | Scale", combined)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------------------------
# 6. Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()