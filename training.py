import numpy as np
import cv2
import os

import face_recognition as fr
#print(fr)

test_img = cv2.imread(r'E:\Projects\Test_image.jpg')


faces_detected,gray_img=fr.faceDetection(test_img)
print("Face Detected: ",faces_detected)


#Training
faces,faceID=fr.labels_for_training_data(r'E:\Projects\images')
face_recognizer=fr.trainClassifier(faces,faceID)
face_recognizer.save(r'E:\Projects\trainingData.yml')

name={0:'Yash',1:'Diksha',2:'Yana'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)    
    print(label)
    print(confidence)
    fr.draw_rect(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows