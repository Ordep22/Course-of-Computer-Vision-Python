import cv2 as cv

#Importantodo os parâmetros de classificação
detectaRelegios = cv.CascadeClassifier("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Visão Computacional/2_Detectação de Faces/Haarcascade/clocks.xml")

#Lendo a imagem
imagemOriginal  = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Visão Computacional/2_Detectação de Faces/Images/clock.jpg")

#Convertendo a imagem lida para escala de cinza
imagemCinza = cv.cvtColor(imagemOriginal,cv.COLOR_BGR2GRAY)


#Identificando os objetos
relogiosIdentificados = detectaRelegios.detectMultiScale(imagemCinza,scaleFactor=1.03,minNeighbors=1)

print(relogiosIdentificados)

for x1,y1,x2,y2 in relogiosIdentificados:

    cv.rectangle(imagemOriginal,(x1,y1),(x1+x2,y1+y2),(0,255,0),1)




cv.imshow("Identificacao de Relogios",imagemOriginal)
cv.waitKey()