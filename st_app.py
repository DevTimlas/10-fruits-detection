import streamlit as st
import torch
from PIL import Image, ImageOps
import io, os, time
from datetime import datetime
import io, os

st.title("FruitsFood Detection with TF")
st.header("Up to 10 classes of fruits Detection Test with Tensroflow")
st.text("Upload a Fruit Image to detect and see it's contents.")
st.text("Model not 99% accurate")
RESULT_FOLDER = './10-fruits-detection/'


def make_predict(img_pth, image_file):
    model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt')  # or yolov5m, yolov5l, yolov5x, custom
    model.conf = 0.1
    model.iou = 0.3
    size = (480, 480)
    img = ImageOps.fit(img_pth, size, Image.ANTIALIAS)
    
    ts = datetime.timestamp(datetime.now())
    imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
    outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
    print(outputpath)
    with open(imgpath, mode="wb") as f:
        f.write(image_file.getbuffer())

    results = model(img)
    results.render()  # render bbox in image
    
    for im in results.ims:
        im_base64 = Image.fromarray(im)
        im_base64.save(outputpath)
        
    info2 = results.pandas().xyxy[0]
    for v in info2['name']:
        st.write(v)
        
    img_ = Image.open(outputpath)
    st.image(img_, caption='Model Prediction(s)', use_column_width='always')
    # st.write(f"{info2.name.to_string(index=False)} \n")
    #results.save(RESULT_FOLDER)
    # predicted_image = Image.open('./fruitsfooddetection/static/image0.jpg')
    #st.write(os.getcwd())
    #results.save(RESULT_FOLDER)
    #predicted_image = Image.open('/app/10-fruits-detection/image0.jpg')
    #st.image(predicted_image, caption="Predicted Image", use_column_width=False)
    # st.write(results.pandas().xyxy[0].to_json(orient='records'))


uploaded_file = st.file_uploader("Choose a face Image ...", type=["jpg", "png", "jpeg", "GIF"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Fruit Image.', use_column_width=False)
    st.write("")
    st.write("predictions ...")
    make_predict(image, uploaded_file)
