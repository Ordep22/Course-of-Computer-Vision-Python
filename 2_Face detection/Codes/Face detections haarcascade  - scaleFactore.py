
import cv2 as cv  ## Importanto as images

img = cv.imread(r'/Users/PedroVitorPereira/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Projetos-Python/Visão Computacional/2_Detectação de Faces/Images/people1.jpg')

##Em visão computacional é muito interessante reformatar o tamanho da imagem
#Para isso podemos usar os comando resize
#Reajustando a imagem
img_ajustada   = cv.resize(img,(800,600))

#Convertendo a imgem para escala de sinza
img_escalacinza = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#Informar para a biblioteca qual o local do arquivo que contem as informações
#do algoritmo treinado
detectorFacial  = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

#Podemos também ajustar a identificação das faces por meio do parametro scalefactori
#Esse parâmetro deve ser maior que um. Lembrando que quanto maior, menor será sua eficácia.

deteccoes  = detectorFacial.detectMultiScale(img_escalacinza,scaleFactor= 1.35)

'''
Como é possível observar, o valor para o scaleFctor mais adequado à 
este caso seia o 1.35. Pois, com esse valor é possível identificar as 
cinco faces frontais. 
'''

#Detectando as faces a partir da imagem RGB gerda
#deteccoes  = detectorFacial.detectMultiScale(img_escalacinza)

#Laço que percorre toda a lisa de informações detectadas
for x1, y1, x2, y2 in deteccoes:

    #Os paâmetros da função rectangle são
    #(IMAGEM, POSIÇAO X E Y DE INICIO, O TAMANHO QUE SERA DEZENHADO NA IMAGEM, A COR EM RGB, A EXPESSURA DO RETANGULO)
    cv.rectangle(img ,(x1,y1),(x1+x2,y1+y2),(0,255,255),5)


##A função imshow requer dois parâmetro o nome o qual a janela irá receber e qual a iagem que está mstrando
cv.imshow("Detectcao das faces!",img)
cv.waitKey()




