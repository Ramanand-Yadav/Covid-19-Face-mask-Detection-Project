import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import datetime


mymodel=load_model('mask_detector.h5')

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    success, img=cap.read()
    
    face=face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)
        
        test_image = image.load_img('temp.jpg', target_size = (150,150,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        
        
        pred = mymodel.predict(test_image)[0][0]
        
        if pred==1:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.rectangle(img, (x, y-40), (x+w, y), (0,0,255), -1)
            cv2.putText(img, "NO MASK", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)    
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0),2)
            cv2.rectangle(img, (x, y-40), (x+w, y), (0,255,0), -1)
            cv2.putText(img, "MASK", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()