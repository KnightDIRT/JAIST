import cv2

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    raise IOError("Cannot open camera")

# Window setup
cv2.namedWindow("Camera Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Control", 640, 480)

# Helper functions
def nothing(x):
    pass

# Create trackbars for camera parameters
cv2.createTrackbar("Exposure", "Camera Control", 5, 20, nothing)         # negative scale factor
cv2.createTrackbar("Gain", "Camera Control", 0, 30, nothing)
cv2.createTrackbar("Brightness", "Camera Control", 0, 100, nothing)
cv2.createTrackbar("Contrast", "Camera Control", 10, 100, nothing)
cv2.createTrackbar("WB Temp", "Camera Control", 4500, 10000, nothing)
cv2.createTrackbar("Gamma", "Camera Control", 10, 300, nothing)
cv2.createTrackbar("AutoExp", "Camera Control", 0, 1, nothing)
cv2.createTrackbar("AutoWB", "Camera Control", 0, 1, nothing)

print("🎥 Use sliders to adjust camera parameters. Press 'q' to quit.")

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        break

    # Read slider values
    exposure_slider = cv2.getTrackbarPos("Exposure", "Camera Control")
    gain = cv2.getTrackbarPos("Gain", "Camera Control")
    brightness = cv2.getTrackbarPos("Brightness", "Camera Control")
    contrast = cv2.getTrackbarPos("Contrast", "Camera Control")
    wb_temp = cv2.getTrackbarPos("WB Temp", "Camera Control")
    gamma_slider = cv2.getTrackbarPos("Gamma", "Camera Control")
    auto_exp = cv2.getTrackbarPos("AutoExp", "Camera Control")
    auto_wb = cv2.getTrackbarPos("AutoWB", "Camera Control")

    # Convert slider scales
    exposure_value = -float(exposure_slider)  # negative exposure scale (for many UVCs)
    gamma = gamma_slider / 100.0 if gamma_slider > 0 else 1.0

    # Apply camera settings
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exp else 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_AUTO_WB, auto_wb)
    if not auto_wb:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp)

    # Apply gamma correction (software-level)
    frame_float = frame.astype('float32') / 255.0
    frame_gamma = cv2.pow(frame_float, gamma)
    frame_display = (frame_gamma * 255).astype('uint8')

    # Display
    cv2.imshow("Camera Control", frame_display)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
