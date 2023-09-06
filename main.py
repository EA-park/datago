import io
import json

from PIL import Image, ImageDraw
import base64

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import torch
import uvicorn


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
app = FastAPI()

origins = [
    "http://34.64.98.254:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main():
    return {"status": "ok"}


class Base64Image(BaseModel):
    type: str
    data: str


@app.post("/detect")
async def detect(image_base64: Base64Image):
    data = image_base64.data.replace("data:image/jpeg;base64,", "")

    # base64 -> binary -> Image object
    data = base64.b64decode(data)
    with io.BytesIO(data) as stream:
        image = Image.open(stream)

        # object detection
        results = model(image)
        results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

        # draw boxes
        draw = ImageDraw.Draw(image)
        for result in results_json:
            xmin, ymin, xmax, ymax = result['xmin'], result['ymin'], result['xmax'], result['ymax']
            draw.rectangle((xmin, ymin, xmax, ymax), outline=(0, 255, 0), width=2)

        # Image object -> binary -> base64
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            image_binary = output.getvalue()
        boxing_image = base64.b64encode(image_binary)

    response = {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({'message': boxing_image})
    }
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
