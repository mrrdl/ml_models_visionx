import pickle
import numpy as np
import cv2
from typing import Tuple

with open('models/face_shape_classification.pkl','rb') as f:
    face = pickle.load(f)

with open('models/pose_measurements.pkl','rb') as f:
    pose = pickle.load(f)

with open('models/skin_tone_detector.pkl','rb') as f :
    skin = pickle.load(f)

def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def predict_face_shape(image: np.ndarray) -> str:
    # Dummy preprocess — replace with actual preprocessing
    processed_img = cv2.resize(image, (224, 224)).flatten().reshape(1, -1)
    return face.predict(processed_img)[0]

def predict_pose_measurements(image: np.ndarray) -> dict:
    # Dummy preprocess — replace with actual MediaPipe logic
    processed_img = cv2.resize(image, (224, 224)).flatten().reshape(1, -1)
    result = pose.predict(processed_img)[0]  # Assuming it returns a list/tuple
    return {
        'height_in': result[0],
        'shoulder_width_in': result[1],
        'chest_width_in': result[2]
    }

def predict_skin_tone(image: np.ndarray) -> str:
    # Dummy preprocess — replace with actual preprocessing
    processed_img = cv2.resize(image, (224, 224)).flatten().reshape(1, -1)
    return skin.predict(processed_img)[0]