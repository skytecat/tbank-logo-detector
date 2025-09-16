from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import io

# Получаем путь к weights/tbank_best.pt относительно app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "weights", "tbank_best.pt")

app = FastAPI()
model = YOLO(model_path) 

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return DetectionResponse(detections=[])

    # Предсказание с порогом уверенности 0.3
    results = model(img, conf=0.3)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(Detection(bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)))

    return DetectionResponse(detections=detections)
