import torch
from flask import Flask, request
import argparse
from PIL import Image
import os
import io

application = Flask(__name__)

RESULT_FOLDER = './fruitsfooddetection/static'
application.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.load('ultralytics/yolov5', 'custom', './fruitsfooddetection/best.pt', force_reload=True)
model.conf = 0.1
model.eval()


@app.route('/', methods=['POST'])
def predict():
    if not request.method == "POST":
        return

    if request.files.get('image'):
        image_file = request.files['image']
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=480)

        results.save(RESULT_FOLDER)

        return results.pandas().xyxy[0].to_json(orient='records')


if __name__ == '__main__':
    app.run()
