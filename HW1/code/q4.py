# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:16:25 2021

@author: hatam
"""


import numpy as np
import cv2 
import matplotlib.pyplot as plt

#read image
img = cv2.imread('retina.png')  

#**********************************************************************
#power law method with gamma>1
my_dpi = 800
fig = plt.figure(figsize=(6, 6), dpi=my_dpi)

# ============ AX1 ============ 
# PIL Image
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("orginal")
ax1.set_xticks([])
ax1.set_yticks([])
pil_img = cv2.imread('retina.png') 
ax1.imshow(pil_img)

# ============ AX2 ============ 
# mpimg image
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("power law with gamma=1.5")
ax2.set_xticks([])
ax2.set_yticks([])
max_img=np.max(img)
gamma=1.5
img_pow=max_img*((img/max_img)**gamma)
img_pow=img_pow.astype('uint8')
ax2.imshow(img_pow)

# ============ AX3 ============ 
# CV2 image (default)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("power law with gamma=2")
ax3.set_xticks([])
ax3.set_yticks([])
max_img=np.max(img)
gamma=2
img_pow=max_img*((img/max_img)**gamma)
img_pow=img_pow.astype('uint8')
ax3.imshow(img_pow)


# ============ AX4 ============ 
# CV2 image (transform)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("power law with gamma=3")
ax4.set_xticks([])
ax4.set_yticks([])
max_img=np.max(img)
gamma=3
img_pow=max_img*((img/max_img)**gamma)
img_pow=img_pow.astype('uint8')
ax4.imshow(img_pow)

#**********************************************************************
#power law method with gamma<1
my_dpi = 800
fig = plt.figure(figsize=(6, 6), dpi=my_dpi)

# ============ AX1 ============ 
# PIL Image
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("orginal")
ax1.set_xticks([])
ax1.set_yticks([])
pil_img = cv2.imread('retina.png') 
ax1.imshow(pil_img)

# ============ AX2 ============ 
# mpimg image
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("power law with gamma=0.8")
ax2.set_xticks([])
ax2.set_yticks([])
max_img=np.max(img)
gamma=0.8
img_pow=max_img*((img/max_img)**gamma)
img_pow=img_pow.astype('uint8')
ax2.imshow(img_pow)

# ============ AX3 ============ 
# CV2 image (default)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("power law with gamma=0.5")
ax3.set_xticks([])
ax3.set_yticks([])
max_img=np.max(img)
gamma=0.5
img_pow=max_img*((img/max_img)**gamma)
img_pow=img_pow.astype('uint8')
ax3.imshow(img_pow)


# ============ AX4 ============ 
# CV2 image (transform)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("power law with gamma=0.3")
ax4.set_xticks([])
ax4.set_yticks([])
max_img=np.max(img)
gamma=0.3
img_pow=max_img*((img/max_img)**gamma)
img_pow=img_pow.astype('uint8')
ax4.imshow(img_pow)

#**********************************************************************
# Apply log transformation method 
#c = 255 / np.log(1 + np.max(img)) 

my_dpi = 800
fig = plt.figure(figsize=(6, 6), dpi=my_dpi)

# ============ AX1 ============ 
# PIL Image
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("orginal")
pil_img = cv2.imread('retina.png') 
ax1.imshow(pil_img)

# ============ AX2 ============ 
# mpimg image
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Logarithmic Transformation")
ax2.set_xticks([])
ax2.set_yticks([])
max_img=np.max(img)
c=np.max(img)/(np.log(np.max(img)-1))
log_img = c * (np.log(img + 1)) 


log_img=log_img.astype('uint8') 
ax2.imshow(log_img)