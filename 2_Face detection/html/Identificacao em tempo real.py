import cv2 as cv
import numpy as np
import imutils

# Abrindo os parâmetro de para identificação do rosto
identificadorFaces = cv.CascadeClassifier("/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/Haarcascade/haarcascade_frontalface_default.xml")

# Iniciando a captura das imagens pela web can
capturaImagens = cv.VideoCapture(0)
capturaImagens.set(3,640) # set Width
capturaImagens.set(4,480) # set Height

while True:

    # Recebendo o frame e o status da captura
    status, frame = capturaImagens.read()

    # Transformando o frama para escala de cinza
    frameCinza = cv.cvtColor(frame[105:550,500:850], cv.COLOR_BGR2GRAY)


    # Obtendo a localização das faces detectadas

    try:
        facesIdentificadas = identificadorFaces.detectMultiScale(frameCinza,scaleFactor=1.2, minNeighbors=5, minSize=(150,150))
        print(facesIdentificadas)
        for x1, y1, x2, y2 in facesIdentificadas:
            cv.rectangle(frameCinza, (x1, y1), (x1 + y1, x2 + y2), (0, 255, 0), 1)


    except:
        print("Erro")
        pass

    cv.imshow("Identificacao em tempo real", frameCinza)
    key = cv.waitKey(5)
    if key == 27:
        break


capturaImagens.release()
cv.destroyWindow(None)
