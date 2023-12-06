# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:55:26 2021

@author: hatam
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt

#read image
img = cv2.imread('retina.png')  


my_dpi = 800
fig = plt.figure(figsize=(6, 6), dpi=my_dpi)

# ============ AX1 ============ 
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("orginal")
ax1.set_xticks([])
ax1.set_yticks([])
pil_img = cv2.imread('retina.png') 
ax1.imshow(pil_img)

# ============ AX2 ============ 
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("treshhold without denoising ")
ax2.set_xticks([])
ax2.set_yticks([])


img1=img.copy()
x , y = np.where(img1[:,:,0]<=205)
img1[x,y,:]=0
ax2.imshow(img1)

# ============ AX3 ============ 

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("treshhold with box filter")
ax3.set_xticks([])
ax3.set_yticks([])


kernel = np.array([[1,1,1], [1,1,1],[1,1,1]])
kernel= kernel/9
img2=cv2.filter2D(img, -1, kernel)   
x , y = np.where(img2[:,:,0]<=205)
img2[x,y,:]=0

ax3.imshow(img2)

# ============ AX4 ============ 

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("filtered image after add background")
ax4.set_xticks([])
ax4.set_yticks([])

img2=img2+(1/5)*(img-img2)
x , y = np.where(img2[:,:,0]>=255)
img2[x,y,:]=255
img2=img2.astype('uint8')
ax4.imshow(img2)