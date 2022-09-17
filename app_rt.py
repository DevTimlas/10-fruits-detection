import cv2
import mediapipe as mp
from streamlit_webrtc import *
import streamlit as st
import os


def make_predict(img_pth):
    model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt')  # or yolov5m, yolov5l, yolov5x, custom
    model.conf = 0.1
    model.iou = 0.3
    size = (480, 480)
    img = ImageOps.fit(img_pth, size, Image.ANTIALIAS)
    # img = (io.BytesIO(img))
    results = model(img)
    info2 = results.pandas().xyxy[0]
    for v in info2['name']:
        st.write(v)
    # st.write(f"{info2.name.to_string(index=False)} \n")
    results.save(RESULT_FOLDER)
    predicted_image = Image.open('./fruitsfooddetection/static/image0.jpg')
    st.image(predicted_image, caption="Predicted Image", use_column_width=False)
    # st.write(results.pandas().xyxy[0].to_json(orient='records'))

def livestream():
	class VideoProcessor():
		def recv(self, frame):
			arr = frame.to_ndarray(format="bgr24")
			
			pred = make_predict(arr)
			
			return av.VideoFrame.from_ndarray(pred, format="bgr24")
		
	webrtc_streamer(key="example",
					rtc_configuration={"iceServers":[
					
					{"urls":"stun:openrelay.metered.ca:80",},
					
					{"urls":"turn:openrelay.metered.ca:80",
					"username":"openrelayproject",
					"credential":"openrelayproject",},
					
					{"urls":"turn:openrelay.metered.ca:443",
					"username":"openrelayproject",
					"credential":"openrelayproject",},
					
					{"urls":"turn:openrelay.metered.ca:443?transport=tcp",
					"username":"openrelayproject",
					"credential":"openrelayproject",},],},
					
					video_processor_factory=VideoProcessor,
					media_stream_constraints={"video":True, "audio":False})
					
					
livestream()
