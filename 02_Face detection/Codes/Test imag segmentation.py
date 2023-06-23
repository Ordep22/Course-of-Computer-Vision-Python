import cv2 as cv


imagem  = cv.imread("/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/Images/car.jpg")
imagemSegmentada = imagem[0:200,0:200]

cv.imshow("imagem segmentada",imagemSegmentada)
cv.waitKey()
cv.imshow("Imagem original",imagem)
cv.waitKey()
