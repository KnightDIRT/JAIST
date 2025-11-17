import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add Depth Anything metric repo to path
repo_path = r"C:/Users/Torenia/Depth-Anything/metric_depth"
sys.path.insert(0, repo_path)

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# --------------------------------------------------
# 1. Paths and setup
# --------------------------------------------------
pretrained_resource = "local::C:/Users/Torenia/OneDrive/Documents/GitHub/JAIST/Depth Anything Triangulation/Code/models/depth_anything_metric_depth_indoor.pt"
img_path = "C:/Users/Torenia/OneDrive/Documents/GitHub/JAIST/Depth Anything Triangulation/Code/SavedResults/20251105_154915_Inpainted.png"

# --------------------------------------------------
# 2. Build model manually (no dataset needed)
# --------------------------------------------------
# Get config only to build the architecture
config = get_config("zoedepth", "eval", "nyu", pretrained_resource=pretrained_resource)
# The dataset name "nyu" is *only* used to set image normalization; 
# it won’t require any data files if we skip dataloaders.

model = build_model(config)
model = model.cuda().eval()

# --------------------------------------------------
# 3. Load and preprocess your image
# --------------------------------------------------
image = Image.open(img_path).convert("RGB")

# Resize to typical ZoeDepth input size
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])
image_t = transform(image).unsqueeze(0).cuda()

# --------------------------------------------------
# 4. Run inference
# --------------------------------------------------
with torch.no_grad():
    pred = model(image_t)
    if isinstance(pred, dict):
        pred = pred.get("metric_depth", pred.get("out"))

depth_map = pred.squeeze().cpu().numpy()

# --------------------------------------------------
# 5. Display and save result
# --------------------------------------------------
plt.imshow(depth_map, cmap='magma')
plt.title("Predicted Depth")
plt.colorbar(label="Depth (m)")
plt.show()

plt.imsave("C:/Users/Torenia/OneDrive/Documents/GitHub/JAIST/Depth Anything Triangulation/Code/metric_depth_photo.png", depth_map, cmap='magma')