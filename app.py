import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from pathlib import Path
import tempfile
from Face_Data import face_data

@st.cache_resource
def load_models():
    model_path     = Path.cwd().joinpath('Models')
    faceProto      = str(model_path / "opencv_face_detector.pbtxt")
    faceModel      = str(model_path / "opencv_face_detector_uint8.pb")
    ageProto       = str(model_path / "age_deploy.prototxt")
    ageModel       = str(model_path / "age_net.caffemodel")
    genderProto    = str(model_path / "gender_deploy.prototxt")
    genderModel    = str(model_path / "gender_net.caffemodel")
    faceNet        = cv2.dnn.readNet(faceModel,faceProto)
    ageNet         = cv2.dnn.readNet(ageModel,ageProto)
    genderNet      = cv2.dnn.readNet(genderModel,genderProto)
    model_mask     = load_model(str(model_path / 'mask_detection.h5'))
    model_disease  = load_model(str(model_path / 'skin_disease.h5'))
    return model_disease, model_mask, genderNet, ageNet, faceNet

model_disease, model_mask, genderNet, ageNet, faceNet = load_models()

# Start Streamlit
st.title('Face Analysis App')


model = lambda image: face_data(faceNet,
                                genderNet,
                                ageNet,
                                model_mask,
                                model_disease,
                                image)

# Let the user upload an image
st.sidebar.header("Upload Image")
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)
    old_image_height, old_image_width, channels = image.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = 3 * old_image_width
    new_image_height = 3 * old_image_height
    color = (255,255,255)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
    
    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    
    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = image
    image = model(result)
    st.image(image, caption="Uploaded Image.", use_column_width=True)



# Let the user upload a video
st.sidebar.header("Upload Video")
uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
FRAME_WINDOW = st.image([])
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())

    video_file = cv2.VideoCapture(tfile.name)
    while video_file.isOpened():
        ret, frame = video_file.read()
        if not ret:
            break
        old_image_height, old_image_width, channels = frame.shape
    
        # create new image of desired size and color (blue) for padding
        new_image_width = 2 * old_image_width
        new_image_height = 2 * old_image_height
        color = (255,255,255)
        result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2
        
        # copy img image into center of result image
        result[y_center:y_center+old_image_height, 
            x_center:x_center+old_image_width] = frame
        frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512))
        frame = model(frame)
        try:
            FRAME_WINDOW.image(frame)
        except:
            pass                        

    video_file.release()

# Webcam input (this requires opencv)
st.sidebar.header("Webcam")
run_webcam = st.sidebar.button('Start Webcam')
FRAME_WINDOW = st.image([]) 
if run_webcam:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
            frame = model(frame)
            FRAME_WINDOW.image(frame)
        except:
            pass

    cap.release()
