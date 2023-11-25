import cv2
import numpy as np
from tensorflow.keras.applications import VGG19

class_mode = ["Acne",
            "Eczema",
            "Atopic",
            "Psoriasis",
            "Tinea",
            "vitiligo"]
vgg_model = VGG19(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3)) 

def skin_disease_detection(image, model):    
    # Preprocess
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
    image = cv2.resize(image, (180, 180))        
    image = np.expand_dims(image, axis=0)
    
    # Feature Extraction
    fe = vgg_model.predict(image)
    fe = fe.reshape(1,-1)   
    
    # Predict
    pred = model.predict(fe)[0]
    
    return class_mode[np.argmax(pred)]
