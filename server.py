# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference import read_image_from_bytes, predict_face_shape, predict_pose_measurements, predict_skin_tone
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the VisionX API"}

@app.post("/predict/face-shape")
async def face_shape_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)
    label = predict_face_shape(image)
    return {"face_shape": label}

@app.post("/predict/pose-measurement")
async def pose_measurement_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)
    measurements = predict_pose_measurements(image)
    return measurements

@app.post("/predict/skin-tone")
async def skin_tone_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)
    tone = predict_skin_tone(image)
    return {"skin_tone": tone}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

