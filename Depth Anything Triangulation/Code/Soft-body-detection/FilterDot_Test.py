import cv2
import numpy as np

# Parameters
AREA_THRESHOLD = 1000         # max area in pixels
CIRCULARITY_THRESHOLD = 0.75 # circularity filter
CENTER_RADIUS = 300          # pixels from center

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    center = (width // 2, height // 2)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color mask
    lower_white = np.array([0, 0, 155])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean mask
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    # Find contours
    contours_info = cv2.findContours(mask_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    # Draw small contours (area < AREA_THRESHOLD) only
    small_contours_frame = frame.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= AREA_THRESHOLD:
            cv2.drawContours(small_contours_frame, [cnt], -1, (255, 0, 0), 2)

    # Draw filtered contours (small + circular + near center)
    filtered_frame = frame.copy()
    cv2.circle(filtered_frame, center, CENTER_RADIUS, (0, 255, 255), 2)  # center reference

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > AREA_THRESHOLD:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < CIRCULARITY_THRESHOLD:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if np.hypot(cX - center[0], cY - center[1]) > CENTER_RADIUS:
            continue

        cv2.drawContours(filtered_frame, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(filtered_frame, (cX, cY), 5, (0, 0, 255), -1)

    # Prepare 2x2 display
    mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
    top_row = np.hstack((frame, mask_bgr))
    bottom_row = np.hstack((small_contours_frame, filtered_frame))
    combined = np.vstack((top_row, bottom_row))

    # Resize if too large
    combined = cv2.resize(combined, (0,0), fx=0.5, fy=0.5)

    cv2.imshow("Camera Analysis 2x2", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
