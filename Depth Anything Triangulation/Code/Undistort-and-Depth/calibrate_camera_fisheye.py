# -*- coding: utf-8 -*-
"""
Fisheye camera calibration with robust handling of ill-conditioned images,
and real-time undistortion on live camera feed.
"""

import cv2
import numpy as np
import glob
import json

# === PARAMETERS ===
CHECKERBOARD = (10, 7)  # 10x7 inner corners
CALIBRATION_IMAGES = "C:/Users/Torenia/OneDrive/Pictures/Camera Roll/Camera_Calibration3/*.jpg"
SAVE_JSON = "fisheye_calibration_data.json"


# === FUNCTIONS ===
def calibrate_fisheye(all_image_points, all_true_points, image_size):
    """
    Robust fisheye calibration that removes ill-conditioned images.
    :param all_image_points: list of (N,2) arrays of detected corners
    :param all_true_points:  list of (N,3) arrays of true 3D points
    :param image_size: (width, height) of images
    :return: rms, K, D, rvecs, tvecs
    """
    assert len(all_true_points) == len(all_image_points), "Mismatch in number of image/true point sets"
    all_true_points = list(all_true_points)
    all_image_points = list(all_image_points)

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
             cv2.fisheye.CALIB_CHECK_COND +
             cv2.fisheye.CALIB_FIX_SKEW)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    while True:
        assert len(all_true_points) > 0, "No valid images left for calibration!"

        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=[p.reshape(1, -1, 3) for p in all_true_points],
                imagePoints=[q.reshape(1, -1, 2) for q in all_image_points],
                image_size=image_size,
                K=np.zeros((3, 3)),
                D=np.zeros((4, 1)),
                flags=flags,
                criteria=criteria
            )
            print(f"✅ Calibration succeeded with {len(all_true_points)} images.")
            return rms, K, D, rvecs, tvecs

        except cv2.error:
            print("⚠️ Calibration failed. Trying to drop ill-conditioned images...")

            removed = False
            for i in range(len(all_true_points)):
                try:
                    test_true = all_true_points[:i] + all_true_points[i+1:]
                    test_img = all_image_points[:i] + all_image_points[i+1:]

                    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                        objectPoints=[p.reshape(1, -1, 3) for p in test_true],
                        imagePoints=[q.reshape(1, -1, 2) for q in test_img],
                        image_size=image_size,
                        K=np.zeros((3, 3)),
                        D=np.zeros((4, 1)),
                        flags=flags,
                        criteria=criteria
                    )
                    print(f"🗑️ Removed image {i}, calibration succeeded with {len(test_true)} images.")
                    all_true_points, all_image_points = test_true, test_img
                    removed = True
                    break
                except cv2.error:
                    continue

            if not removed:
                raise RuntimeError("❌ Could not find a valid subset of images for calibration")


# === MAIN SCRIPT ===

# Prepare object points for checkerboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob(CALIBRATION_IMAGES)
print(f"Found {len(images)} images")

_img_shape = None
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if _img_shape is None:
        _img_shape = gray.shape[::-1]
    else:
        assert _img_shape == gray.shape[::-1], "All images must have the same size"

    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (3, 3), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )
        objpoints.append(objp.copy())
        imgpoints.append(corners2.reshape(-1, 2))

print(f"Using {len(objpoints)} valid images for calibration")

# Run calibration
rms, K, D, rvecs, tvecs = calibrate_fisheye(imgpoints, objpoints, _img_shape)

print("RMS:", rms)
print("K:", K)
print("D:", D)

# Save calibration
data = {
    'image_size': _img_shape,
    'K': K.tolist(),
    'D': D.tolist(),
    'rms': rms
}
with open(SAVE_JSON, "w") as f:
    json.dump(data, f)
print(f"Calibration saved to {SAVE_JSON}")

# === REAL-TIME CAMERA UNDISTORT ===
cap = cv2.VideoCapture(0)  # Open default camera

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

# Prepare undistortion maps
while True:
    ret, frame = cap.read()
    if not ret:
        break

    dim1 = frame.shape[:2][::-1]  # (width, height)

    scaled_K = K * dim1[0] / _img_shape[0]
    scaled_K[2][2] = 1.0

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, D, dim1, np.eye(3), balance=1.0
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2
    )

    undistorted_frame = cv2.remap(
        frame, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # Show side by side
    combined = np.hstack((frame, undistorted_frame))
    cv2.imshow("Original (left) vs Undistorted (right)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
