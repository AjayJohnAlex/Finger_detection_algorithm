import numpy as np 
import matplotlib.pyplot as plt
import cv2
%matplotlib inline


def load_pic():
    img = cv2.imread('lion.jpeg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def display_img(img,cmap=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    plt.imshow(img)


i = load_pic()
pic = display_img(i,cmap='gray')

lion_pic = cv2.imread('lion.jpeg',0)
plt.imshow(lion_pic,cmap='gray')

ret , thresh1 = cv2.threshold(lion_pic,125,250,cv2.THRESH_BINARY)
plt.imshow(thresh1,cmap='gray')

# thresholding
ret1 , thresh2 = cv2.threshold(lion_pic,125,250,cv2.THRESH_BINARY_INV)
plt.imshow(thresh1,cmap='gray')

ret2 , thresh = cv2.threshold(lion_pic,125,250,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(thresh1,cmap='gray')


# Blurring
lion = load_pic()
filter_lion = cv2.filter2D(lion,-1,kernel)
plt.imshow(filter_lion)


# Image Gradient 
sobelx = cv2.Sobel(pic_new,cv2.CV_64F,1,0,ksize=5)
plt.imshow(sobelx,cmap='gray')

lap_lion = cv2.Laplacian(pic_new,cv2.CV_64F,ksize=5)
plt.imshow(lap_lion,cmap='gray')

blended = cv2.addWeighted(sobelx,0.5,sobely,0.7,0)
plt.imshow(blended,cmap='gray')


img_lion = cv2.imread('lion.jpeg')
color = ['b','g','r']

for i , col in enumerate(color):
    histLi = cv2.calcHist([img_lion],[i],None,[251],[0,251])
    plt.plot(histLi,color=col)
    plt.xlim([0,251])