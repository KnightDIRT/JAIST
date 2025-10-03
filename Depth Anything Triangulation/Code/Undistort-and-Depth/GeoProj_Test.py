"""
GeoProj realtime viewer with CUDA rectification fallback to CPU.

- Tries: from resample.resampling import rectification (CUDA/numba)
- If rectification(...) raises an exception (cuda context error etc),
  it falls back to rectification_cpu(...) implemented here.

Usage:
    python GeoProj_realtime_fallback.py
Press 'q' to quit.
"""

import sys
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# --- change repo_path as needed ---
repo_path = r"C:/Users/Torenia/GeoProj"
sys.path.insert(0, repo_path)

# try import CUDA rectification from repo
use_cuda_rectify = False
try:
    from resample.resampling import rectification as rectification_cuda
    use_cuda_rectify = True
    print("Found repo rectification (CUDA). Will try CUDA version first.")
except Exception as e:
    rectification_cuda = None
    print("Repo rectification import failed or not present; will use CPU fallback. Import error:", e)


# --- Model imports ---
from modelNetM import EncoderNet, DecoderNet, ClassNet

# --- Transformation same as eval.py ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Load models ---
model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()

if torch.cuda.device_count() > 1:
    model_en = nn.DataParallel(model_en)
    model_de = nn.DataParallel(model_de)
    model_class = nn.DataParallel(model_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_en.to(device); model_de.to(device); model_class.to(device)

# load pretrained weights (non-strict to tolerate "module." prefix)
model_en.load_state_dict(torch.load(f"{repo_path}/model_en.pkl", map_location=device), strict=False)
model_de.load_state_dict(torch.load(f"{repo_path}/model_de.pkl", map_location=device), strict=False)
model_class.load_state_dict(torch.load(f"{repo_path}/model_class.pkl", map_location=device), strict=False)

model_en.eval(); model_de.eval(); model_class.eval()

types = ['barrel','pincushion','rotation','shear','projective','wave']

# -------------------------
# CPU fallback rectification
# -------------------------
# Implements iterative fixed-point search:
# For each target pixel t=(xr,yr) solve for source s=(i,j) such that s + f(s) = t
# using iteration:
#   s_{k+1} = t - f(s_k)   where f(s) is sampled by bilinear interpolation of forward flow.
#
# If converged and inside image bounds, sample distorted image at s (bilinear),
# else mark mask as invalid (255).

def bilinear_sample_flow(flow, x, y):
    """
    flow: np.array shape (2,H,W) float32
    x,y: floats (x: col, y: row)
    returns (u,v) as floats
    """
    H = flow.shape[1]
    W = flow.shape[2]
    if x < 0 or x >= W-1 or y < 0 or y >= H-1:
        # Out of range: clamp to nearest valid
        xi = min(max(int(np.round(x)), 0), W-1)
        yi = min(max(int(np.round(y)), 0), H-1)
        return float(flow[0, yi, xi]), float(flow[1, yi, xi])
    i = int(np.floor(x))
    j = int(np.floor(y))
    dx = x - i
    dy = y - j

    # four neighbors
    u11 = flow[0, j, i]; v11 = flow[1, j, i]
    u12 = flow[0, j, i+1]; v12 = flow[1, j, i+1]
    u21 = flow[0, j+1, i]; v21 = flow[1, j+1, i]
    u22 = flow[0, j+1, i+1]; v22 = flow[1, j+1, i+1]

    u = u11*(1-dx)*(1-dy) + u12*(dx)*(1-dy) + u21*(1-dx)*(dy) + u22*(dx)*(dy)
    v = v11*(1-dx)*(1-dy) + v12*(dx)*(1-dy) + v21*(1-dx)*(dy) + v22*(dx)*(dy)
    return float(u), float(v)

def bilinear_sample_img(img, x, y):
    """
    img: HxWx3 uint8 or HxW grayscale
    x: col, y: row (float)
    returns interpolated pixel as tuple or scalar
    """
    H, W = img.shape[:2]
    if x < 0 or x >= W-1 or y < 0 or y >= H-1:
        # outside: return white (255)
        if img.ndim == 3:
            return (255,255,255)
        else:
            return 255

    i = int(np.floor(x))
    j = int(np.floor(y))
    dx = x - i
    dy = y - j

    if img.ndim == 3:
        Q11 = img[j, i].astype(np.float32)
        Q12 = img[j, i+1].astype(np.float32)
        Q21 = img[j+1, i].astype(np.float32)
        Q22 = img[j+1, i+1].astype(np.float32)
        val = Q11*(1-dx)*(1-dy) + Q12*(dx)*(1-dy) + Q21*(1-dx)*(dy) + Q22*(dx)*(dy)
        val = np.clip(val, 0, 255).astype(np.uint8)
        return (int(val[0]), int(val[1]), int(val[2]))
    else:
        Q11 = float(img[j, i])
        Q12 = float(img[j, i+1])
        Q21 = float(img[j+1, i])
        Q22 = float(img[j+1, i+1])
        val = Q11*(1-dx)*(1-dy) + Q12*(dx)*(1-dy) + Q21*(1-dx)*(dy) + Q22*(dx)*(dy)
        return int(np.clip(val, 0, 255))

def rectification_cpu(distorted, flow, max_iter=60, precision=1e-2):
    """
    distorted: HxWx3 uint8
    flow: 2xHxW float32  (forward flow)
    returns: (corrected_img (HxWx3 uint8), mask (HxW uint8) where 0 valid, 255 invalid)
    """
    H, W = distorted.shape[0], distorted.shape[1]
    result = np.ones_like(distorted, dtype=np.uint8) * 255
    mask = np.zeros((H, W), dtype=np.uint8)  # 0 valid, 255 invalid

    # pad flow to avoid indexing edge issues (we'll clamp inside sampler)
    # iterate per pixel (target pixel coordinates xr,yr)
    for yr in range(H):
        for xr in range(W):
            # initial guess: do a single-step inverse approx using local flow at integer location
            # sample flow at (xr,yr) integer location
            fx = int(min(max(xr, 0), W-1))
            fy = int(min(max(yr, 0), H-1))
            u0 = float(flow[0, fy, fx])
            v0 = float(flow[1, fy, fx])
            i = xr - u0
            j = yr - v0

            converged = False
            for _it in range(max_iter):
                # sample flow at current guess s=(i,j)
                u_s, v_s = bilinear_sample_flow(flow, i, j)
                i_next = xr - u_s
                j_next = yr - v_s

                if (abs(i_next - i) < precision) and (abs(j_next - j) < precision):
                    i, j = i_next, j_next
                    converged = True
                    break
                i, j = i_next, j_next

                # if guess moves outside reasonable bounds early, break
                if i < 0 or i >= W-1 or j < 0 or j >= H-1:
                    converged = False
                    break

            if converged and 0 <= i < W-1 and 0 <= j < H-1:
                # sample distorted image at (i,j)
                pix = bilinear_sample_img(distorted, i, j)
                if distorted.ndim == 3:
                    result[yr, xr, 0] = pix[0]
                    result[yr, xr, 1] = pix[1]
                    result[yr, xr, 2] = pix[2]
                else:
                    result[yr, xr] = pix
                # mask remains 0 (valid)
            else:
                mask[yr, xr] = 255  # invalid

    return result, mask

# -------------------------
# End CPU rectification
# -------------------------

# --- Start camera loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index 0).")

frame_skip = 0            # set >0 to process every (frame_skip+1)-th frame (keep UI smoother)
frame_count = 0

print("Starting capture. Press 'q' to quit.")
use_cuda = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # optionally skip frames for speed
    if frame_skip > 0 and (frame_count % (frame_skip + 1) != 0):
        cv2.imshow("GeoProj - Before/After", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # prepare input
    distorted = cv2.resize(frame, (256, 256))
    img_tensor = transform(distorted).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        middle = model_en(img_tensor)
        flow_output = model_de(middle)
        clas = model_class(middle)
        _, predicted = torch.max(clas.data, 1)
        label = types[predicted.item()]

        flow = flow_output[0].cpu().numpy().astype(np.float32)  # shape (2,H,W)

    # print some flow stats
    fmin = float(np.min(flow)); fmax = float(np.max(flow)); fmean = float(np.mean(flow))
    print(f"Flow stats min/max/mean: {fmin:.4f} / {fmax:.4f} / {fmean:.4f}")

    # try CUDA rectification first (if available)
    corrected = None
    mask = None
    t_rect0 = time.time()
    if use_cuda_rectify:
        try:
            corrected, mask = rectification_cuda(distorted, flow)
            used_method = "cuda_repo"
        except Exception as e:
            print("CUDA rectification failed at runtime (falling back to CPU). Error:", e)
            try:
                corrected, mask = rectification_cpu(distorted, flow)
                used_method = "cpu_fallback"
            except Exception as e2:
                print("CPU fallback rectification also failed:", e2)
                corrected = np.ones_like(distorted) * 255
                mask = np.ones((distorted.shape[0], distorted.shape[1]), dtype=np.uint8) * 255
                used_method = "failed"
    else:
        # directly use CPU fallback
        corrected, mask = rectification_cpu(distorted, flow)
        used_method = "cpu_fallback"

    t_rect1 = time.time()

    # overlay label and info
    cv2.putText(distorted, f"Pred: {label}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(corrected, f"Method: {used_method}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(corrected, f"RectT: {t_rect1-t_rect0:.2f}s", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # show mask small
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    combined = np.hstack((distorted, corrected, mask_bgr))
    # enlarge window for easier viewing
    combined_view = cv2.resize(combined, (combined.shape[1]*2, combined.shape[0]*2))

    cv2.imshow("GeoProj - Original | Corrected | Mask", combined_view)

    t1 = time.time()
    print(f"Frame total time: {t1 - t0:.2f}s  Rectify time: {t_rect1 - t_rect0:.2f}s  Method: {used_method}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
