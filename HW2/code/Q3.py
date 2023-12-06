# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:59:08 2021

@author: hatam
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

img = cv2.imread('chessboard.jpg')
'''section A'''
kernel = np.array([[1,-1],[0,0]])
kernel1 = np.array([1,0])
kernel2= np.array([[0,0,0],[1,0,-1],[0,0,0]]) #equal [1 , 0 ,-1]
kernel3 = np.array([[0,1,0],[0,0,0],[0,-1,0]])
kernel4= np.array([[-1 , -1,-1],[-1,8,-1],[-1,-1,-1]])

img1 = cv2.filter2D(img, -1, kernel)
img2 = cv2.filter2D(img, -1, kernel1)
img3= cv2.filter2D(img, -1, kernel2)
img4 = cv2.filter2D(img, -1, kernel3)
img5 = cv2.filter2D(img, -1, kernel4)

#grad = signal.convolve2d(img, kernel3, boundary='symm', mode='same')
#plt.imshow(img1)
#plt.imshow(img2)
#plt.imshow(img3)
#plt.imshow(img4)
#plt.imshow(grad)


'''section B'''

'''cany'''
edges = cv2.Canny(img,100,300)

'''sobel'''
img_grey= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(img_grey,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img_grey,cv2.CV_64F,0,1,ksize=5)  # y

'''x=np.where(sobelx<0)
sobelx[x]=0
x=np.where(sobelx>255)
sobelx[x]=255
x=np.where(sobely<0)
sobely[x]=0
x=np.where(sobely>255)
sobely[x]=255
#cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0, sobel);
sobelx=sobelx/(sobelx.max()-sobelx.min())
sobely=sobely/(sobely.max()-sobely.min())
sobel_grey=0.5*sobelx+0.5*sobely
sobel_grey=sobel_grey*255
sobel_grey=sobel_grey.astype("uint8")
'''
ads_x=cv2.convertScaleAbs(sobelx);
ads_y=cv2.convertScaleAbs(sobely);
sobel_grey=0.5*ads_x+0.5*ads_y

#sobel=  cv2.cvtColor(sobel_grey,cv2.COLOR_GRAY2RGB)



'''LOG'''

# Apply Gaussian Blur
blur = cv2.GaussianBlur(img_grey,(5,5),0)

# Apply Laplacian operator in some higher datatype
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
#laplacian=laplacian/(laplacian.max()-laplacian.min())
x=np.where(laplacian<0)
laplacian[x]=0
x=np.where(laplacian>255)
laplacian[x]=255
laplacian=laplacian.astype("uint8")



plt.subplot(131)
plt.imshow(edges,cmap = 'gray')
plt.title('canny'), plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(sobel_grey,cmap = 'gray')
plt.title('sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(laplacian,cmap = 'gray')
plt.title('laplacian'), plt.xticks([]), plt.yticks([])