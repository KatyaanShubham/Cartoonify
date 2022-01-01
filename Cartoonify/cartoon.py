import cv2
import numpy as np


def read_image(image_name):
    img = cv2.imread(image_name)
    return img

def img_edge(img,edge_width,blur):

    #convert color image to gray scale
    gC = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #covert gray scale image to blur image
    gB = cv2.medianBlur(gC,blur)

    #calculate and store the image edges
    iE= cv2.adaptiveThreshold(gB,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,edge_width,blur)
    return iE

def quantization(img,k):
    iD = np.float32(img).reshape((-1,3))
    iC = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret, label, center = cv2.kmeans(iD,k,None,iC,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    iN = center[label.flatten()]
    iN = iN.reshape(img.shape)
    return iN




image = read_image(r".path to image\image_name.imagetype")
edge_width = 9
blur_value = 9
totalColors = 10 # play with this number to achieve your desired output

img_Edge = img_edge(image,edge_width,blur_value)
image = quantization(image,totalColors)
blurred = cv2.bilateralFilter(image,d=7,sigmaColor=200,sigmaSpace=200)
cartoonify = cv2.bitwise_and(blurred,blurred,mask=img_Edge)
cv2.imwrite(".\Cartoon_Image\converted.jpeg",cartoonify)