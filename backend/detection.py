import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

VISIBILITY_THRESHOLD = 0.6

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='pose.task'),
        running_mode=vision.RunningMode.IMAGE
    )
)

face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE
    )
)

def extract_features_from_frame(frame):

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    if not pose_result.pose_landmarks:
        return None

    if not face_result.face_landmarks:
        return None

    landmarks = pose_result.pose_landmarks[0]

    left_ear = landmarks[7]
    right_ear = landmarks[8]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    if not (
        left_shoulder.visibility > VISIBILITY_THRESHOLD and
        right_shoulder.visibility > VISIBILITY_THRESHOLD and
        left_ear.visibility > VISIBILITY_THRESHOLD and
        right_ear.visibility > VISIBILITY_THRESHOLD
    ):
        return None

    # Midpoints
    ear_x = (left_ear.x + right_ear.x) / 2
    ear_y = (left_ear.y + right_ear.y) / 2
    ear_z = (left_ear.z + right_ear.z) / 2

    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    shoulder_z = (left_shoulder.z + right_shoulder.z) / 2

    # Eye distance normalization
    face_landmarks = face_result.face_landmarks[0]
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]

    eye_distance = math.hypot(
        left_eye.x - right_eye.x,
        left_eye.y - right_eye.y
    )

    if eye_distance <= 1e-6:
        return None

    # === FEATURES (same as training) ===
    dx = ear_x - shoulder_x
    dy = shoulder_y - ear_y
    side_angle = math.degrees(math.atan2(dx, dy))

    forward_ratio_z = (shoulder_z - ear_z) / eye_distance
    vertical_offset = (shoulder_y - ear_y) / eye_distance
    shoulder_slope = (right_shoulder.y - left_shoulder.y) / eye_distance

    return [
        side_angle,
        forward_ratio_z,
        vertical_offset,
        shoulder_slope
    ]