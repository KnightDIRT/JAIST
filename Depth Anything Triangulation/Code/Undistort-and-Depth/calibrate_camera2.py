#Dot

import cv2
import numpy as np
import glob

# === SETTINGS ===
pattern_size = (5, 3)   # (columns, rows) -> from gen_pattern.py
square_size = 15.0       # mm

# Collect all calibration images
images = glob.glob("C:/Users/Torenia/OneDrive/Pictures/Camera Roll/Camera_Calibration3/*.jpg")

# Prepare object points in real world space
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
# For asymmetric circle grid: x = (2 * column + row % 2), y = row
objp[:, :2] = np.array([
    [(2 * c + r % 2) * square_size, r * square_size]
    for r in range(pattern_size[1])
    for c in range(pattern_size[0])
])

objpoints = []  # 3D points
imgpoints = []  # 2D points

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# === PROCESS IMAGES ===
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Look for asymmetric circle grid
    ret, centers = cv2.findCirclesGrid(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ASYMMETRIC_GRID
    )

    if ret:
        objpoints.append(objp)
        # refine detections
        centers2 = cv2.cornerSubPix(gray, centers, (5,5), (-1,-1), criteria)
        imgpoints.append(centers2)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, centers2, ret)
        cv2.imshow("Asymmetric Circles Detection", vis)
        cv2.waitKey(500)
    else:
        print(f"❌ Circles not found in {fname}")

cv2.destroyAllWindows()

print(f"Collected {len(objpoints)} valid images for calibration.")

# === CALIBRATION ===
if len(objpoints) > 0:
    flags = cv2.CALIB_RATIONAL_MODEL  # higher order distortion

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags
    )

    print("Calibration RMS error:", ret)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    np.save("camera_matrix.npy", camera_matrix)
    np.save("dist_coeffs.npy", dist_coeffs)
else:
    print("❌ No valid patterns detected. Try better lighting or check rows/cols config.")
