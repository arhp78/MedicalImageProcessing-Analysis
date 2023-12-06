# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:00:41 2021

@author: hatam
"""

import numpy as np
import cv2 

def rgbtogray(img_BGR):
    gray=np.zeros_like(img[:,:,0])
    gray=0.299 * img_BGR[:,:,2] + 0.587 * img_BGR[:,:,1] + 0.114 *img_BGR[:,:,0]
    return gray
img=cv2.imread("Brain_MRI.png")
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray_3_channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
Hori = np.concatenate((img, img_gray_3_channel), axis=1) 
#cv2.imshow('brain',Hori) 

# section b
imgeg_rgbtogray=rgbtogray(img)
imgeg_rgbtogray1=imgeg_rgbtogray.astype("uint8")
cv2.imshow('brain_gray_python',img_gray) 
cv2.imshow('brain_gray',imgeg_rgbtogray1) 
#Gray = np.concatenate((imgeg_rgbtogray, img_gray), axis=1) 
#cv2.imshow('brain_gray',Gray) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

cv2.imwrite("Brain_MRI_grayscale.png",imgeg_rgbtogray1 )