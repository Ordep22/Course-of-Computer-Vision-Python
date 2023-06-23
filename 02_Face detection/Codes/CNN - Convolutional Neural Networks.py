import dlib
import cv2 as cv

image  = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Images/people2.jpg")

cnnFacedetection = dlib.cnn_face_detection_model_v1("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Weights/mmod_human_face_detector.dat")

detections  = cnnFacedetection(image,1)

#print(detections)


for faces in detections:
    left,top,right,bottom, trust = faces.rect.left(), faces.rect.top(), faces.rect.right(), faces.rect.bottom(), faces.confidence
    print(trust)
    cv.rectangle(image,(left,top),(right, bottom),(0,255,255),2)




cv.imshow("Face detection with CNN",image)
cv.waitKey()

