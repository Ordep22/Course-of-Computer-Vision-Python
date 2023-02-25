import cv2 as cv

#Classificador da imagem
#Detector olhos
detectorOlhos  = cv.CascadeClassifier("/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/haarcascade_eye.xml")

#Detector facial
detectorFacil = cv.CascadeClassifier("/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/haarcascade_frontalface_default.xml")

#Abrindo a imagem
img  = cv.imread("/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/Images/people1.jpg")

#Convertendo a imagem em escala de cinza
imgCinza = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


#Detecao das faces
deteccoesFaces  = detectorFacil.detectMultiScale(imgCinza,scaleFactor= 1.09,minNeighbors=10,minSize=(35,35),maxSize=(150,150))
# x1 & y1 posicionamento x2 & y2 tamanho
for (x1,y1,x2,y2) in deteccoesFaces:
     cv.rectangle(img,(x1,y1),(x1+x2,y1+y2),(0,255,0),2)

#Deteccão dos olhos
deteccoesOlhos = detectorOlhos.detectMultiScale(imgCinza,scaleFactor= 1.09,minNeighbors=10,minSize=(15,15),maxSize=(70,70))

# x1 & y1 posicionamento x2 & y2 tamanho
for (x1,y1,x2,y2) in deteccoesOlhos:
    cv.rectangle(img,(x1,y1),(x1+x2,y1+y2),(0,0,255),2)
    print(x2,y2)

'''
É possível perceber que em alguns casos não será possível identificar 
corretamente os componentes da imagem, seja por resolução ou qualquer coutro motivo 


'''

cv.imshow("Deteccaoo de Olhos e faces",img)
cv.waitKey()

'''
Desafio criar um for dentro de cada face e identifcar o olhos que estão contidos 
naquele local
'''








