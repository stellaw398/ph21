#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:45:36 2018

@author: apple
"""
from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

img1 = Image.open('chameleon.jpeg')
plt.figure(0)
plt.imshow(img1)
plt.show()
print img1.size
plt.figure(1)
img2 = img1.filter(ImageFilter.FIND_EDGES)
plt.imshow(img2)
plt.savefig('FindEdges.pdf')
plt.close()

##blurring
img= misc.imread('chameleon.jpeg','RGB')
blur = np.zeros(img.shape)
ndimage.gaussian_filter(img,sigma =5 ,output=blur)
plt.figure(2)
plt.imshow(blur)
plt.savefig('blur.pdf')
plt.close()


plt.figure(3)
cnt = 1
for i in np.arange(0.5,2.5,0.5):
    edge = ndimage.gaussian_gradient_magnitude(img, sigma=i)
    plt.subplot(2,2,cnt)
    plt.imshow(edge)
    plt.xlabel("width = %.2f"%i)
    cnt +=1
plt.tight_layout()
plt.savefig('widthtest.pdf')
plt.close() 

edge2 = ndimage.generic_gradient_magnitude(img, derivative= ndimage.sobel)
plt.figure(4)
plt.imshow(edge2)
plt.savefig('Sobel.pdf')
plt.close()
edge3 = ndimage.generic_gradient_magnitude(img, derivative= ndimage.prewitt)
plt.figure(5)
plt.imshow(edge3)
plt.savefig('Prewitt.pdf')
plt.close()