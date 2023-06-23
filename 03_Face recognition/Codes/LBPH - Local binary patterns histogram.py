import os
import  cv2 as cv
import numpy as np
from PIL import Image



def GETIMAGEDATA():
    listPaths = []
    faces  = []
    ids  = []

    dirPath = r"/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Course-of-Computer-Vision-Python/03_Face recognition/Datasets/yalefaces/train"


    #Obtain the path of images
    for itens in os.listdir(dirPath):

        #Join the dir path and images paths
        path = os.path.join(dirPath,itens)

        #Includ paths in a list
        listPaths.append(path)

        #Convert PIL imagens in Numpy array
        image = Image.open(path).convert('L')

        image_np = np.array(image,'uint8')

        ids.append(int(itens[7:9:1]))
        faces.append(image_np)

    return np.asarray(ids), np.asarray(faces)


ids, faces = GETIMAGEDATA()

#print(type(faces))
#print(type(faces[0]))

#Training LBPH with images for train
lbphClassifier  = cv.face.LBPHFaceRecognizer_create()
lbphClassifier.train(faces, ids)
lbphClassifier.write("lbphClassifier.yml")

#Reconiza imagens from test directori

#Crate the obeject of a class
lbphFaceclassifier = cv.face.LBPHFaceRecognizer_create()

#Openc tranine file
lbphFaceclassifier.read("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Course-of-Computer-Vision-Python/03_Face recognition/Codes/lbphClassifier.yml")

#Path from aleatori image
pathImgetest = "/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Course-of-Computer-Vision-Python/03_Face recognition/Datasets/yalefaces/test/subject07.happy.gif"

#Open image in gray scale
imageTest  = Image.open(pathImgetest).convert("L")

#Convert the image from image to Np array
imagetestNparray  = np.asarray(imageTest,"uint8")

#Criate the prediction of a imgem recognaze
predition = lbphFaceclassifier.predict(imagetestNparray)

#Show prediction
#print(predition)

#Show image and result found
cv.putText(imagetestNparray,"Pred:" + str(predition[0]) , (10,30),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
cv.putText(imagetestNparray,"Exp:" + pathImgetest[142:144], (10,50),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
cv.imshow("Image recognize",imagetestNparray)
cv.waitKey()






















