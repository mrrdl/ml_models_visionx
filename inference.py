import pickle
import numpy as np
import cv2
import torch
import tensorflow as tf
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import mediapipe as mp
import tempfile

# Load face shape & skin tone models
face = torch.load('models/face_shape_classification.pth', map_location=torch.device('cpu'), weights_only=False)
face.eval()

skin = tf.keras.models.load_model('models/skin_tone_detector.h5')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True)

# Helper: Convert image bytes to OpenCV format
def read_image_cv2(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# For face/skin classifier
def preprocess_for_classification(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor.numpy().reshape(1, -1)

# Face shape classifier
face_class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

def predict_face_shape(image_bytes: bytes) -> str:
    tensor = preprocess_for_classification(image_bytes, as_tensor=True)
    with torch.no_grad():
        logits = face(tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
    return face_class_names[predicted_class]

# Skin tone classifier
def preprocess_for_skin_tone(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((200, 200))
    image_array = np.array(image) / 255.0  # normalize
    return np.expand_dims(image_array, axis=0)  # shape: (1, 200, 200, 3)

def predict_skin_tone(image_bytes: bytes) -> str:
    tensor = preprocess_for_skin_tone(image_bytes)
    prediction = skin.predict(tensor)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    class_labels = ['white', 'brown', 'black']  # update this based on your dataset
    return class_labels[predicted_class]

# Pose measurement using MediaPipe (no .pkl)
def predict_pose_measurements(image_bytes: bytes) -> dict:
    # Write to temporary file for OpenCV
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        image_path = tmp.name

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)

    if not results.pose_landmarks:
        raise ValueError("No pose landmarks found.")

    h, w, _ = img.shape
    landmarks = results.pose_landmarks.landmark

    def get_coords(landmark): return np.array([landmark.x * w, landmark.y * h])

    left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
    right_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    left_hip = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
    right_hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
    nose = get_coords(landmarks[mp_pose.PoseLandmark.NOSE])
    left_ankle = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    chest_width = np.linalg.norm(left_hip - right_hip)
    torso_height = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
    approx_height = np.linalg.norm(left_ankle - nose)

    return {
        "height_in": round(approx_height / 37.8, 2),            # assuming 96 DPI (~1 px = 1/96 in)
        "shoulder_width_in": round(shoulder_width / 37.8, 2),
        "chest_width_in": round(chest_width / 37.8, 2),
        "torso_height_in": round(torso_height / 37.8, 2)
    }
