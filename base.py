import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('casecade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels ={}
with open("labels.pickle",'rb') as file: #rb for : 
    labels = pickle.load(file) #pickle.load for # dump information to that file
    labels= {v:k for k,v in labels.items()} #invert

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the frame to go gray 
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces : # where my face is
        print(x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w] #take only the square of gray chnages de dimenstion : y:y+h, x:x+w
        
        #recognizer
        id_,conf = recognizer.predict(roi_gray) #give the id of the predictable person
        if conf>=45 and conf <=85 :
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color=(255,255,225)
            stroke=2
            cv2.putText(frame, name, (x,y), font , 1, color , stroke , cv2.LINE_AA)
        img_item= "img.png" 
        cv2.imwrite(img_item,roi_gray) #save image
        color=(255,0,0) #BGR 
        stroke= 2 # how thick the line will be
        width= x+w
        height = y+h
        cv2.rectangle(frame, (x,y), (width, height), color, stroke) #dirli moustatil de  corre=doné x et y et de h et w 3erd et toul

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()