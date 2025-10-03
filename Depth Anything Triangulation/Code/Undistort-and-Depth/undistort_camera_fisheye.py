import cv2
import numpy as np
import json

# Load calibration results from JSON
with open("fisheye_calibration_data.json", "r") as f:
    calib_data = json.load(f)

K = np.array(calib_data["K"])
D = np.array(calib_data["D"])
image_size = tuple(calib_data["image_size"])

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

# Precompute undistortion maps for efficiency
new_K = K.copy()
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2
)

print("✅ Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize camera frame to match calibration resolution if needed
    frame_resized = cv2.resize(frame, image_size)

    # Undistort using precomputed maps
    undistorted = cv2.remap(frame_resized, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Show before/after side by side
    combined = np.hstack((frame_resized, undistorted))

    # 🔹 Scale down by 1/2
    display = cv2.resize(combined, (combined.shape[1] // 2, combined.shape[0] // 2))

    cv2.imshow("Fisheye Calibration Test (Left: Raw | Right: Undistorted)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
