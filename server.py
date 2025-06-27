
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import (
    predict_face_shape,
    predict_pose_measurements,
    generate_fashion_prompt,
    get_recommmendation,
    get_imgid,
    generate_url,
    fetch_image_from_url,
    image_to_base64,
    get_image_json
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
        face= predict_face_shape(image_bytes)
        pose = predict_pose_measurements(image_bytes)

        return {
            "face_shape": face,
            **pose,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get-img")
async def get_id(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        face= predict_face_shape(image_bytes)
        pose = predict_pose_measurements(image_bytes)

        data= {
            "face_shape": face,
            **pose,
        }
        prompt = generate_fashion_prompt(data)
        recommendation = get_recommmendation(prompt)
        img_id=get_imgid(recommendation)
        img_url= generate_url(img_id)
        img_json=get_image_json(img_url)
        return{
            "img_json":img_json,
            "img_id": str(img_id),
            "img_url": img_url,
            "prompt": prompt,
            "recommendation": recommendation,
            "data":data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
