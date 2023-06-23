import dlib
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as  plt
from PIL import  Image




#haarcascade

originalImageHaarcascade = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Images/people3.jpg")

grayScaleimeg  = cv.cvtColor(originalImageHaarcascade ,cv.COLOR_BGR2GRAY)

faceDetection = cv.CascadeClassifier("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Haarcascade/haarcascade_frontalface_default.xml")

detections = faceDetection.detectMultiScale(grayScaleimeg,scaleFactor=1.001,minNeighbors=7,minSize=(5,5))

for (x1,y1,x2,y2 ) in detections :

    cv.rectangle(originalImageHaarcascade,(x1,x2),(x1+x2,y1+y2),(0,0,255),2)



#Convert the original image in a array
originalImageHaarcascadearray = np.asarray(cv.cvtColor(originalImageHaarcascade, cv.COLOR_BGR2RGB))



#HOG

originalImageHog = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Images/people3.jpg")

#In this method is necessary input tow argument , They`re the image and scale of the image. This argument is similar to scale factore in Haarcascade
faceDetectionhog = dlib.get_frontal_face_detector() #Sharch why I don`t input the argumente in this parte

hogDetections = faceDetectionhog(originalImageHog,4)

for elements in hogDetections:
    left,top,right,bottom = elements.left(),elements.top(),elements.right(),elements.bottom()
    cv.rectangle(originalImageHog,(left,top),(right,bottom),(0,0,255),2)




#Convert the original image in a array
originalImageHogarray = np.asarray(cv.cvtColor(originalImageHog, cv.COLOR_BGR2RGB))

#CNN

image  = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Images/people3.jpg")

#Weigths
cnnFacedetection = dlib.cnn_face_detection_model_v1("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Weights/mmod_human_face_detector.dat")

detections  = cnnFacedetection(image,4)

#print(detections)


for faces in detections:
    left,top,right,bottom, trust = faces.rect.left(), faces.rect.top(), faces.rect.right(), faces.rect.bottom(), faces.confidence
    print(trust)
    cv.rectangle(image,(left,top),(right, bottom),(0,0,255),2)



#Convert the original image in a array
originalImageCnnarray = np.asarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))





plt.subplot(3,1,1)
plt.title("Haarcascade result")
plt.imshow(originalImageHaarcascadearray)
plt.axis("off")

plt.subplot(3,1,2)
plt.title("HOG result")
plt.imshow(originalImageHogarray)
plt.axis("off")

plt.subplot(3,1,3)
plt.title("CNN result")
plt.imshow(originalImageCnnarray)
plt.axis("off")


plt.show()
plt.waitforbuttonpress(0)






