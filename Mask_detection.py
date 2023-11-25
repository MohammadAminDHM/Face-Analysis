import cv2
import numpy as np
class_mode = ['mask_weared_incorrect',
            'with_mask',
            'without_mask']
def mask_detection(image, model):    
    # Preprocess
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
    image = cv2.resize(image, (224, 224))        
    image = np.expand_dims(image, axis=0)
    
    # Predict
    out_1 = model.predict(image)    
    return class_mode[np.argmax(out_1, axis=1)[0]]