import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# ===== Beep function (cross-platform) =====
def beep():
    try:
        import winsound
        winsound.Beep(1000, 200)
    except:
        os.system('printf "\\a"')

# ===== Angle calculation =====
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# ===== MediaPipe setup =====
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Prevent constant beeping
last_beep_time = 0
BEEP_COOLDOWN = 1.0  # seconds

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get right arm points
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Display angle
            cv2.putText(image, f'Elbow Angle: {int(angle)}',
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            # ===== Form Check =====
            # Ideal shooting elbow angle ~ 70°–110°
            if angle < 70 or angle > 110:
                cv2.putText(image, "⚠ Bad Form: Elbow Misaligned",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)

                current_time = time.time()
                if current_time - last_beep_time > BEEP_COOLDOWN:
                    beep()
                    last_beep_time = current_time

            else:
                cv2.putText(image, "✓ Good Form",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 3)

        except:
            pass

        # Draw skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

        cv2.imshow('NeuroShot Prototype', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()