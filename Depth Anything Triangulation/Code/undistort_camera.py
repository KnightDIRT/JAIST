import cv2
import numpy as np

# Load calibration
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# Open camera
cap = cv2.VideoCapture(1)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get size 
    h, w = frame.shape[:2]

    # Compute optimal new camera matrix (optional, improves FOV)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistort frame
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Show side by side
    combined = np.hstack((frame, undistorted))
    cv2.imshow("Original (left) vs Undistorted (right)", combined)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
