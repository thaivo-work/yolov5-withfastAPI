from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status , File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from loguru import logger
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import io
from starlette.responses import Response

from detection_model import get_yolov5, get_image_from_bytes

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1"
    )
security = HTTPBasic()

model = get_yolov5()

origins = [
    "http://localhost",
    "http://localhost:8081",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="png")
    return Response(content=bytes_io.getvalue(), media_type="image/png")