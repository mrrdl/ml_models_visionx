import pickle
import numpy as np
import datetime
import ast
import pandas as pd
import cv2
import torch
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import mediapipe as mp
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
import base64
from fastapi.responses import JSONResponse
from io import BytesIO

import os
genai.configure(api_key="AIzaSyD9gdqwwdyzH2BV146Vy3ETosOoTIETKLQ")

model=genai.GenerativeModel(model_name="gemini-1.5-flash")

# Load face shape & skin tone models
face = torch.load('models/face_shape_classification.pth', map_location=torch.device('cpu'), weights_only=False)
face.eval()

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
    return tensor

# Face shape classifier
face_class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

def predict_face_shape(image_bytes: bytes) -> str:
    tensor = preprocess_for_classification(image_bytes)
    with torch.no_grad():
        logits = face(tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
    return face_class_names[predicted_class]

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


def generate_fashion_prompt(user_input: dict) -> str:
    face_shape = user_input.get("face_shape", "Oval")
    height = round(user_input.get("height_in", 18.5), 1)
    shoulder = round(user_input.get("shoulder_width_in", 4.0), 1)
    chest = round(user_input.get("chest_width_in", 2.3), 1)
    torso = round(user_input.get("torso_height_in", 6.7), 1)

    return (
        f"Recommend a stylish outfit for men with a {face_shape.lower()} face shape, "
        f"shoulders {shoulder} inches wide, chest width {chest} inches, and "
        f"torso height of {torso} inches."
    )

def get_recommmendation(prompt: str) -> str:
    output=model.generate_content(prompt)
    return output.text.strip()

def get_imgid(prompt: str) -> int:
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_style_text(text: str) -> np.ndarray:
        return encoder.encode([text])

    df = pd.read_csv("styles.csv",on_bad_lines='skip')

    texts = df["productDisplayName"].astype(str).tolist()

    embeddings = encoder.encode(texts)

    # embeddings is already a numpy array of shape (num_rows, embedding_dim)
    item_embeddings = embeddings

    def recommend_from_gemini(topk=5):
        style_text = get_recommmendation(prompt)
        emb = embed_style_text(style_text)
        sims = cosine_similarity(emb, item_embeddings)[0]
        idx = np.argsort(sims)[::-1][:topk]
        recommended_ids = df.iloc[idx]["id"].tolist()
        return recommended_ids
    
    recommended_ids = recommend_from_gemini()
    print(f"Recommended Items: {recommended_ids}")
    return recommended_ids[0] if recommended_ids else None

def generate_url(img_id: str) -> str:
    base_url = "https://storage.googleapis.com/visionx-clothes-data/ClothesData/images/"
    return f"{base_url}{img_id}.jpg"

def fetch_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def image_to_base64(image: Image.Image) -> str:
    img_buffer = BytesIO()
    image.save(img_buffer, format="JPEG")  # or PNG
    img_bytes = img_buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def get_image_json(img_url: str) -> JSONResponse:
    image = fetch_image_from_url(img_url)
    image_b64 = image_to_base64(image)

    return JSONResponse(content={
        "image_base64": image_b64
    })


    
