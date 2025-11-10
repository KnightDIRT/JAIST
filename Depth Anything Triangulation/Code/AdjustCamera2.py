import cv2
import numpy as np
import time

def auto_adjust(frame):
    """Balanced tone adjustment — bright, vivid, soft reflections, no grey cast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # --- Adaptive exposure and contrast ---
    alpha = np.clip(1.15 + (128 - mean_brightness) / 700.0, 0.9, 1.6)
    beta = np.clip((130 - mean_brightness) / 2.0, -15, 25)
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # --- HSV space for tone/color refinement ---
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float32)
    s = s.astype(np.float32)

    # Gentle highlight softening (no flattening)
    v = np.where(v > 230, 230 + (v - 230) * 0.9, v)
    v = np.power(v / 255.0, 0.95) * 255.0  # gamma lift for midtones
    v = np.clip(v, 0, 255)

    # Restore color depth (fix grey cast)
    s = np.clip(s * 1.25 + 5, 0, 250)

    hsv = cv2.merge((h, s.astype(np.uint8), v.astype(np.uint8)))
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Gentle warming to reduce cold/grey tone ---
    b, g, r = cv2.split(adjusted)
    r = np.clip(r * 1.05, 0, 255)   # warm tone
    b = np.clip(b * 0.97, 0, 255)
    adjusted = cv2.merge((b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)))

    # --- Subtle smoothing (optional) ---
    adjusted = cv2.bilateralFilter(adjusted, 5, 20, 20)

    return adjusted

def set_auto_mode(cap, enable_auto):
    """Enable or disable the camera's internal auto exposure and white balance."""
    # Wait a bit before applying new settings (prevents crash on some cameras)
    time.sleep(0.2)
    if enable_auto:
        print("[MODE] Default auto mode enabled")
        # Some drivers use 0.75 for auto, 0.25 for manual (DirectShow)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    else:
        print("[MODE] Custom mode enabled (manual exposure/WB)")
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)

# --- Initialize ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # or try CAP_V4L2 on Linux
if not cap.isOpened():
    raise RuntimeError("Camera not found")

use_custom = False
set_auto_mode(cap, use_custom)

print("Press 't' to toggle mode | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed")
        break

    if use_custom:
        output = auto_adjust(frame)
        mode = "Custom Auto-Adjust"
    else:
        output = frame
        mode = "Default Auto Mode"

    cv2.putText(output, mode, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Camera", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        use_custom = not use_custom
        set_auto_mode(cap, not use_custom)

cap.release()
cv2.destroyAllWindows()
