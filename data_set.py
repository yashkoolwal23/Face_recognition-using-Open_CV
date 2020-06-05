#creating dataset of images
import cv2
import sys


cpt=0

vidStream=cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret,frame = vidStream.read()
    # Display the resulting frame
    cv2.imshow("Test Frame",frame)

    cv2.imwrite(r"E:\Projects\images\0\image%04i.jpg" %cpt,frame)
    cpt +=1

    if cv2.waitKey(10)==ord('q'):
        break