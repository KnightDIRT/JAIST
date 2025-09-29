import sys
import cv2
import torch
import numpy as np

# --- Adjust paths ---
repo_path = r"C:/Users/Torenia/SGDepth"  # path to repo root
model_path = r"./models/model.pth"

sys.path.insert(0, repo_path)

from models.sgdepth import SGDepth  # your file is models/sgdepth.py

# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = SGDepth()
    checkpoint = torch.load(model_path, map_location=device)

    # Your checkpoint is already a state_dict
    state = checkpoint  

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

def prepare_frame(frame, height=192, width=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor.to(device)

def depth_to_colormap(disp_tensor):
    disp = disp_tensor.squeeze().cpu().numpy()
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
    return disp_color

def run_realtime():
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = prepare_frame(frame)

        # Wrap into batch format expected by SGDepth
        dataset = {
            ('color_aug', 0, 0): tensor,
            'purposes': [('depth',)],   # tell SGDepth to run depth
        }
        batch = [dataset]

        with torch.no_grad():
            outputs = model(batch)

        # outputs is a tuple of dicts, one per dataset
        out = outputs[0]
        disp = out[('disp', 0)]  # highest resolution disparity

        disp_vis = depth_to_colormap(disp)
        disp_vis = cv2.resize(disp_vis, (frame.shape[1], frame.shape[0]))

        combined = np.hstack((frame, disp_vis))
        cv2.imshow("SGDepth Realtime (RGB | Disp)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
