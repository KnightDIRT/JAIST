import cv2
import numpy as np
import glob
import os

# === Settings ===
CALIBRATION_IMAGES_DIR = "C:/Users/Torenia/OneDrive/Pictures/Camera Roll/Camera_Calibration3"  # folder with checkerboard images
BOARD_SIZE = (10, 7)   # (columns, rows) of inner corners
SQUARE_SIZE = 0.02     # physical size of squares (any unit)
SHOW_CORNERS = True
SAVE_FILE = "fisheye_calibration.npz"

def calibrate_from_folder():
    # prepare 3D points for one board
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float64)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    img_shape = None

    images = glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, "*.jpg"))

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, (3, 3), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

            # ✅ enforce correct shapes
            objpoints.append(objp.reshape(-1, 1, 3))
            imgpoints.append(corners.reshape(-1, 1, 2))

            if SHOW_CORNERS:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, BOARD_SIZE, corners, ret)
                cv2.imshow("Corners", vis)
                cv2.waitKey(200)

    cv2.destroyAllWindows()

    N_OK = len(objpoints)
    print(f"Found corners in {N_OK} images.")

    if N_OK < 3:
        raise RuntimeError("Not enough valid calibration images!")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    # calibration flags
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        image_size=img_shape,
        K=K,
        D=D,
        rvecs=None,
        tvecs=None,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    print("RMS error:", rms)
    print("K:\n", K)
    print("D:\n", D)

    np.savez(SAVE_FILE, K=K, D=D, img_shape=img_shape)
    print(f"Calibration saved to {SAVE_FILE}")


if __name__ == "__main__":
    calibrate_from_folder()
