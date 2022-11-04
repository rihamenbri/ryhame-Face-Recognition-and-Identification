import cv2
import os 
from PIL import Image
import numpy as np
import pickle

#open  the directory this file is  and  collect images and put them in a  list that i willl be training 
#open  the directory this file is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir =os.path.join(BASE_DIR, "images") #images is the directory name where the images are

face_cascade = cv2.CascadeClassifier('casecade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id=0
y_labels =[]
x_train=[]
label_ids={}

for  root, dirs, files in os.walk(image_dir):
    for file in files :
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ", "-")# mettre le nom du fichier comme label de ses photos and you can remplace  "os.path.dirname(path)" with " root" 
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            
            #x_tra in.append(path)
            #y_labels.append(label)
            pil_image= Image.open(path).convert("L") #open image li 3ednha adress "path" and convert to gray 
            image_array= np.array(pil_image,"uint8") #uint8. Unsigned integer (0 to 255) for pixels genralmenet 
            #print(image_array)
            faces= face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
            
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w] #take only the square of gray chnages de dimenstion : y:y+h, x:x+w
                x_train.append(roi) #ajouter le petit square  ala list de train
                y_labels.append(id_)

#print(y_labels)
# print(x_train)
 
with open("labels.pickle",'wb') as file: #wb cause am wiritng bytes
    pickle.dump(label_ids, file) #pickle.dump() function to store the object data to the file.


recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
