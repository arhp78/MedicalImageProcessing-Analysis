# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:02:44 2021

@author: hatam
"""


import numpy as np
import cv2 
import matplotlib.pyplot as plt

img=cv2.imread("Hist.tif")
hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
#plt.plot(hist_img)

plt.plot(hist_img)
plt.show()

#adapthisteq
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
adapthisteq= clahe.apply(img[:,:,0])

hist_adapthisteq = cv2.calcHist([adapthisteq],[0],None,[256],[0,256])

plt.plot(hist_adapthisteq)
plt.show()
cv2.imwrite('adapthisteq.png',adapthisteq)

#equalization
equ = cv2.equalizeHist(img[:,:,0])
hist_equ = cv2.calcHist([equ],[0],None,[256],[0,256])

plt.plot(hist_equ)
plt.show()

cv2.imwrite('histeq.png',equ)
Hori = np.concatenate(( equ,adapthisteq), axis=1) 

cv2.imshow('histeq-adapthisteq',Hori) 
#Gray = np.concatenate((imgeg_rgbtogray, img_gray), axis=1) 
#cv2.imshow('brain_gray',Gray) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

