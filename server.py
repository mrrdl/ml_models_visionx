from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import (
    predict_face_shape,
    predict_pose_measurements,
    predict_skin_tone
)
import uvicorn

app = FastAPI()

# Enable CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/face-shape")
async def face_shape_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_face_shape(image_bytes)
        return {"face_shape": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/skin-tone")
async def skin_tone_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_skin_tone(image_bytes)
        return {"skin_tone": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/pose-measurement")
async def pose_measurement_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_pose_measurements(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    image_bytes = await file.read()

    
    try:
        skin = predict_skin_tone(image_bytes)
        pose = predict_pose_measurements(image_bytes)
        return {
            'skin_tone': skin,
            **pose
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
