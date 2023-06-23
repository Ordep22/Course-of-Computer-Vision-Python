
"""
HOG - Histograns of orientade Gradients

"""

import dlib
import cv2 as cv


file = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Images/people2.jpg")

faceDetesctorhog  = dlib.get_frontal_face_detector()

detections = faceDetesctorhog(file,3) #In this method is necessary input one argument , it's the scale of the image. This argument is similar to scale factore


for faces in detections:
    left,top,right,bottom = faces.left(), faces.top(), faces.right(), faces.bottom()
    cv.rectangle(file,(left,top),(right, bottom),(0,255,255),2)


cv.imshow("Teste",file)
cv.waitKey()






