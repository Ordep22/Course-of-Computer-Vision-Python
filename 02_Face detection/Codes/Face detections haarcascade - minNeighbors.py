import cv2 as cv

#Abrindo a imagem
imagem  = cv.imread("/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/Images/people2.jpg")

#Transformando para escala de cinza
imagemEscalacinza = cv.cvtColor(imagem,cv.COLOR_BGR2GRAY)


# #Mostrando a imagem em escala de cinza
# cv.imshow("`Imagem em escala de cinza",imagemEscalacinza)
# cv.waitKey()


#Passar os parâmetro do algoritmo treinado para detecção de faces
detectorFacial = cv.CascadeClassifier("/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/haarcascade_frontalface_default.xml")

'''
****
Estudar o que esse número exadecimal represeta <CascadeClassifier 0x7f9f9635aeb0>
****
'''

#Detectar as faces da imagem
facesDetectadas = detectorFacial.detectMultiScale(imagemEscalacinza,scaleFactor= 1.21,minNeighbors=3,minSize=(35,35),maxSize=(65,65))
'''
O parâmetro minNeighbors fas com que a função calcule qual entre os X retangulos 
definidos é o mais adequando na identificação da imagem.
Quanto maior esse valor maior é a qualidade dos resultados e a certeza de que o que foi identificado é
realmente uma face

'''

#Imprimir na imagem


for x1,y1,x2,y2 in facesDetectadas:

    print(x2,y2)

    cv.rectangle(imagem,(x1,y1),(x1 + x2,y1+y2),(0,0,255),2)


cv.imshow("Faces detectadas",imagem)
cv.waitKey()


















