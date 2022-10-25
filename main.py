# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:10:44 2022

@author: Tavi
"""

import cv2
import numpy as np
from glob import glob
import random
import ipyplot
from matplotlib import pyplot as plt

base_path = r"C:\Users\Tavi\Desktop\CV1 LAB\Images\\"
file_list =[]
for i in range(4): #clase
    one_class_folder = glob(base_path+str(i)+"/*.jpg")
    file_list.append(one_class_folder)

print(file_list) # printam ce poze am gasit

random.seed(42)
indexes = random.sample(range(0,9), 7)
print(indexes)

train_files = []
test_files = []
for f in file_list:
    t = []
    for el in indexes:
        t.append(f[el])
    train_files.append(t)
    test_files.append([x for x in f if x not in t])
    
print(test_files)
print(np.shape(train_files))    


im = cv2.imread(r"C:\Users\Tavi\Desktop\CV1 LAB\Images\0\3.jpg")
#cv2.imshow("sh", im)
#ipyplot.plot_images(im, max_images = 1, img_width = 456)
plt.imshow(im)
# imagina este citita in BGR si trebuie schimbat spatiul de culoare
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im_rgb)
print(np.shape(im_rgb))

im_rgb_small = cv2.resize(im_rgb, (int(im_rgb.shape[1] * 0.2), int(im_rgb.shape[0] * 0.2)))
im_rgb_small = cv2.blur(im_rgb_small, (11,11))
plt.imshow(im_rgb_small)
print(np.shape(im_rgb_small))
plt.figure()
color = ('r', 'g', 'b')
for i, col in enumerate(color):
    hist = cv2.calcHist([im_rgb_small], [i],None, [256], [0, 256])
    plt.plot(hist, color =col)
    plt.xlim([0, 256])
plt.show()
mask = im_rgb_small[...,0] > 150
plt.figure()
plt.imshow(mask)
def sim_col(v):
    if abs(v[0] - 46) < 70 and abs(v[1] - 163) < 70 and abs(v[2] - 157) < 70:
        return (0,0,0)
    return (255,255,255)

mask2 = im_rgb_small.copy()
for i in range(mask2.shape[1]):
    for j in range(mask2.shape[0]):
        mask2[j,i,:] = sim_col(mask2[j, i, :])
plt.figure()    
plt.imshow(mask2)
        
scale = 1
delta = 0
ddepth = cv2.CV_16S

gray = cv2.cvtColor(im_rgb_small, cv2.COLOR_BGR2GRAY)
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
plt.imshow(grad)