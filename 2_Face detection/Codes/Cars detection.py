import cv2 as cv
#Importanto o algoritmo treinado
carrosIdentificados = cv.CascadeClassifier("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Visão Computacional/2_Detectação de Faces/Haarcascade/cars.xml")

#importando a imagem Original
imagemOriginal  = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Visão Computacional/2_Detectação de Faces/Images/car.jpg")

#Convertendo a imagem em escala de cinza
imagemCinza = cv.cvtColor(imagemOriginal,cv.COLOR_BGR2GRAY)

carroDetectados = carrosIdentificados.detectMultiScale(imagemCinza,scaleFactor=1.03,minNeighbors=5)

for x1,y1,x2,y2 in carroDetectados:
    cv.rectangle(imagemOriginal,(x1,y1),(x1+x2,y1+y2),(0,200,0),1)

cv.imshow("Carroa identificados",imagemOriginal)
cv.waitKey()
