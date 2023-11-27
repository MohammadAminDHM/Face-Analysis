import cv2
import numpy as np
from Mask_detection import mask_detection as md
from Skin_Disease_detection import skin_disease_detection as sdd

def calculate_average_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return np.mean(v)

def increase_brightness(img, value=None):
    try:
        # Calculate average brightness if value is not provided
        if value is None:
            average_brightness = calculate_average_brightness(img)
            # Adjust the value based on average brightness
            # For example, less adjustment if the image is already bright
            if average_brightness > 50:  # Image is bright
                value = max(30 - (average_brightness - 50) // 2, 0)
            else:  # Image is dark
                value = 30

        # Brightness adjustment process
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    except Exception as e:
        print("Error:", e)
        return img

def highlightFace(net, frame, conf_threshold=0.7):
    frame          = np.array(frame)
    frameOpencvDnn = frame.copy()
    frameHeight    = frameOpencvDnn.shape[0]
    frameWidth     = frameOpencvDnn.shape[1]
    blob           = cv2.dnn.blobFromImage(frameOpencvDnn,
                                        1.0,
                                        (300, 300),
                                        [104, 117, 123],
                                        True,
                                        False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def face_data(net_face, net_gender, net_age, net_md, net_sdd, frame):
    frame = np.array(frame)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    padding=10
    
    # Face Detection
    resultImg, faceBoxes = highlightFace(net_face, frame)
    
    if not faceBoxes:
        print("No face detected")
    
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):
                min(faceBox[3] + padding,
                frame.shape[0] - 1), max(0, faceBox[0] - padding)
                :min(faceBox[2] + padding,
                frame.shape[1] - 1)]
        
        # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        # face = list(cv2.split(face))

        # face[0] = clahe.apply(face[0])
        # face = cv2.merge(face)
        # face = cv2.cvtColor(face, cv2.COLOR_LAB2BGR)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
            face[:, :, 0] = cv2.equalizeHist(face[:, :, 0])
            face = cv2.cvtColor(face, cv2.COLOR_YCrCb2BGR)            
        except:
            pass
        face = increase_brightness(face)
        # resultImg[max(0, faceBox[1]-padding):
        #         min(faceBox[3] + padding,
        #         frame.shape[0] - 1), max(0, faceBox[0] - padding)
        #         :min(faceBox[2] + padding,
        #         frame.shape[1] - 1)] = face
        
        # Mask Detection
        maskNetOutput = md(face, net_md)
        
        if maskNetOutput!='with_mask':
            
            # Gender Detection
            blob = cv2.dnn.blobFromImage(face,
                                        1.0,
                                        (227,227),
                                        MODEL_MEAN_VALUES,
                                        swapRB=False)
            net_gender.setInput(blob)
            genderPreds = net_gender.forward()
            gender = genderList[genderPreds[0].argmax()]            
            
            # Age Detection
            net_age.setInput(blob)
            agePreds = net_age.forward()
            age = ageList[agePreds[0].argmax()]            
            
            # Disease Detection
            diseaseNetOutput = sdd(face, net_sdd)
            
            line_height = 30  # Height of each line, adjust as needed.
            
            # Calculate the starting y-coordinate for the first line of text
            start_y = faceBox[1] - 10 - line_height
            
            # First line of text (maskNetOutput)
            cv2.putText(resultImg,
                        f'{maskNetOutput}',
                        (faceBox[0], start_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)
            
            # Second line of text (gender)
            cv2.putText(resultImg,
                        f'{gender}',
                        (faceBox[0], start_y - line_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)
            
            # Third line of text (age)
            cv2.putText(resultImg,
                        f'{age}',
                        (faceBox[0], start_y - line_height * 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)
            
            # Fourth line of text (diseaseNetOutput)
            cv2.putText(resultImg,
                        f'{diseaseNetOutput}',
                        (faceBox[0], start_y - line_height * 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)
            
        else:
            cv2.putText(resultImg,
                        f'{maskNetOutput}',
                        (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,0,0),
                        2,
                        cv2.LINE_AA)            
    return resultImg
