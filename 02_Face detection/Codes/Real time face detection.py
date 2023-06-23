import cv2 as cv


# Haacascade frontal face
identificadorFaces = cv.CascadeClassifier("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Haarcascade/haarcascade_frontalface_default.xml")

#Haarcascade eyes
detectorOlhos   = cv.CascadeClassifier("/Users/PedroVitorPereira/Documents/GitHub/Python_Projects/course-of-Computer-Vision-Python/2_Face detection/Haarcascade/haarcascade_eye.xml")

# Start capture in web can
capturaImagens = cv.VideoCapture(0)


while True:

    # Receiving frame and capture status
    status, img  = capturaImagens.read()

    frame  = img[105:550,500:850]

    # Change de image from RGB to GRAY scale
    frameCinza = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


    # Getting the location of identified objects

    try:


        facesIdentificadas = identificadorFaces.detectMultiScale(frameCinza,scaleFactor=1.1,minNeighbors=10, minSize=(100,100))
        print(facesIdentificadas)
        for x1, y1, x2, y2 in facesIdentificadas:
            cv.rectangle(frame, (x1, y1), (x1 + y1, x2 + y2), (0, 255, 0), 1)
 


        # Deteccão dos olhos
        deteccoesOlhos = detectorOlhos.detectMultiScale(frameCinza, scaleFactor=1.1, minNeighbors=10, minSize=(19, 19), maxSize=(70, 70))
        for (x1, y1, x2, y2) in deteccoesOlhos:
            cv.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 2)
            print(x2, y2)



    except:
        print("Error")
        pass

    cv.imshow("Real time face detection", frame)
    key = cv.waitKey(5)
    if key == 27:
        break


capturaImagens.release()
cv.destroyWindow(None)
