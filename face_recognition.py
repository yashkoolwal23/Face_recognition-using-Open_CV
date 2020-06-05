import numpy
import cv2     
import os       # provides functions for interacting with the operating system


def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)      #grayscale is used to change the image pixels in various shades of gray from rgb  so that it can be identified easily
    face_haar = cv2.CascadeClassifier(r'E:\Softwares\Python\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')   #haarcascade will simply detect face from image
    faces = face_haar.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=3)         #scale factor is used to scale down the intensity of pixel values or intensity so that it can identified easily
    return faces,gray_img


def labels_for_training_data(directory):
    faces=[]
    faceID=[]
#os.walk will simple have 3 tuples as mentioned to iterate through them
    for path,subdirnames,filenames in os.walk(directory):
        #to check if there is any error that name should not start other than i in dataset
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system File")
                continue
            #to load 
            id=os.path.basename(path)               #to get the basename of path i.e last name
            img_path=os.path.join(path,filename)   # to concatenate paths
            print("img_path",img_path)
            print("id",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Not Loaded Properly")
                continue

            #drawing a rectangle on image
            faces_rect,gray_img=faceDetection(test_img)
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID        


#training faces using lbph
def trainClassifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    #model takes every number into array
    face_recognizer.train(faces,numpy.array(faceID))
    return face_recognizer

#drawing the rectangle to name it
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

#putting text in that
def put_text(test_img,label_name,x,y):
    cv2.putText(test_img,label_name,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),3)    

