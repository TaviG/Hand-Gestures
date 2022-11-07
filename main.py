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

# aici incarcam toate imaginile intr-o lista de obiecte de cv2
test_img = [cv2.imread(img) for folder in test_files for img in folder ]
train_img = [cv2.imread(img) for folder in train_files for img in folder ]
test_labels = [int(img.split("\\")[-2]) for folder in test_files for img in folder]
train_labels = [int(img.split("\\")[-2]) for folder in train_files for img in folder]
plt.figure(),plt.imshow(train_img[0])

def bgr_to_rgb(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

test_img_col = list(map(bgr_to_rgb, test_img))
train_img_col = list(map(bgr_to_rgb, train_img))

plt.imshow(test_img_col[0])


def resize(img):
    img_resize = cv2.resize(img,(int(img.shape[1] * 0.2), int(img.shape[0] * 0.2)))
    return img_resize

test_img_col_res = list(map(resize, test_img_col))
train_img_col_res = list(map(resize, train_img_col))

def hand_mask(img):
    mask = img[...,0]>95
    return mask

test_img_col_res_mask = list(map(hand_mask, test_img_col_res))
train_img_col_res_mask = list(map(hand_mask, train_img_col_res))

fig, ax = plt.subplots(1, 8, figsize=(12, 6))
for i in range(8):
    ax[i].imshow(test_img_col_res_mask[i], cmap="gray")
plt.show()

def hand_contour(img, img_orig):
    contur, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arie_max = 0
    rect_max = np.array([0, 0, 0, 0])
    for i in contur:
        temp = cv2.boundingRect(i)
        if temp[2] * temp[3] > arie_max: #temp[2] width temp[3] height
            arie_max = temp[2] * temp[3]
            rect_max = temp
            
    return crop_img(img, rect_max), crop_img(img_orig, rect_max)

def crop_img(img, rect):
    return img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

test_packed = list(map(hand_contour, test_img_col_res_mask, test_img_col_res))
train_packed = list(map(hand_contour, train_img_col_res_mask, train_img_col_res))

test_contours = [i[0] for i in test_packed]
test_orig_crop =  [i[1] for i in test_packed]

train_contours = [i[0] for i in train_packed]
train_orig_crop = [i[1] for i in train_packed]


fig, ax = plt.subplots(1, 8, figsize=(12, 6))
for i in range(8):
    ax[i].imshow(test_contours[i], cmap="gray")
plt.show()


fig, ax = plt.subplots(1, 8, figsize=(12, 6))
for i in range(8):
    ax[i].imshow(test_orig_crop[i], cmap="gray")
plt.show()

fig, ax = plt.subplots(4, 7, figsize=(12, 6))
for i in range(4):
    for j in range(7):
        ax[i][j].imshow(train_orig_crop[i*7+j])


# test

# im = cv2.imread(r"C:\Users\Tavi\Desktop\CV1 LAB\Images\0\3.jpg")
# #cv2.imshow("sh", im)
# #ipyplot.plot_images(im, max_images = 1, img_width = 456)
# plt.imshow(im)
# # imagina este citita in BGR si trebuie schimbat spatiul de culoare
# im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# plt.imshow(im_rgb)
# print(np.shape(im_rgb))

# im_rgb_small = cv2.resize(im_rgb, (int(im_rgb.shape[1] * 0.2), int(im_rgb.shape[0] * 0.2)))
# im_rgb_small = cv2.blur(im_rgb_small, (11,11))
# plt.imshow(im_rgb_small)
# print(np.shape(im_rgb_small))
# plt.figure()
# color = ('r', 'g', 'b')
# for i, col in enumerate(color):
#     hist = cv2.calcHist([im_rgb_small], [i],None, [256], [0, 256])
#     plt.plot(hist, color =col)
#     plt.xlim([0, 256])
# plt.show()
# mask = im_rgb_small[...,0] > 150
# plt.figure()
# plt.imshow(mask)
# def sim_col(v):
#     if abs(v[0] - 46) < 70 and abs(v[1] - 163) < 70 and abs(v[2] - 157) < 70:
#         return (0,0,0)
#     return (255,255,255)

# mask2 = im_rgb_small.copy()
# for i in range(mask2.shape[1]):
#     for j in range(mask2.shape[0]):
#         mask2[j,i,:] = sim_col(mask2[j, i, :])
# plt.figure()    
# plt.imshow(mask2)
        
# scale = 1
# delta = 0
# ddepth = cv2.CV_16S

# gray = cv2.cvtColor(im_rgb_small, cv2.COLOR_BGR2GRAY)
# grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
# grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
# abs_grad_x = cv2.convertScaleAbs(grad_x)
# abs_grad_y = cv2.convertScaleAbs(grad_y)

# grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# plt.imshow(grad)
