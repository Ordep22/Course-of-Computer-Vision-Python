import cv2 as cv

#Abrindo o algoritmo treinado para classificação
classificadore  = cv.CascadeClassifier("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Visão Computacional/2_Detectação de Faces/Haarcascade/fullbody.xml")

#Abrindo a imagem original
imagemOriginal  = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/Visão Computacional/2_Detectação de Faces/Images/people3.jpg")

#Convertendo a imagem em escala de cinza
imgemCinza = cv.cvtColor(imagemOriginal,cv.COLOR_BGR2GRAY)

#Identificando o posicionamento dos objetos identificados
corporIdentificado  = classificadore.detectMultiScale(imagemOriginal,  scaleFactor=1.001,minNeighbors=7,flags=None,minSize=(60,60),maxSize=(150,200))
print(corporIdentificado)

'''
[[ 69  57  95 190]
 [268  86  72 144]
 [327  70  86 172]
 [504 158  30  60]
 [145 111  68 136]]

'''
for x1,y1,x2,y2 in corporIdentificado:
    cv.rectangle(imagemOriginal,(x1,y1),(x1+x2,y1+y2),(0,200,0),1)


cv.imshow("Corpos identificados",imagemOriginal)
cv.waitKey()

