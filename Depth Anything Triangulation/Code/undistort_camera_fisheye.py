import cv2
import numpy as np

CALIB_FILE = "fisheye_calibration.npz"
CAMERA_ID = 0
BALANCE = 0.0   # 0=crop, 1=max FOV
NEW_SIZE = None  

def undistort_live():
    npz = np.load(CALIB_FILE)
    K = npz["K"]
    D = npz["D"]
    img_shape = tuple(npz["img_shape"])  # (w,h)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read from camera")
    h, w = frame.shape[:2]

    if NEW_SIZE is None:
        new_size = (w, h)
    else:
        new_size = NEW_SIZE

    # Compute new camera matrix for the desired output size
    R = np.eye(3)
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_shape, R, balance=BALANCE, new_size=new_size
    )

    # Create undistort maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, P, new_size, cv2.CV_16SC2
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.remap(frame, map1, map2,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow("Undistorted", undistorted)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    undistort_live()
