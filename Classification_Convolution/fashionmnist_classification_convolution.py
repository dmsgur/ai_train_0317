#!/usr/bin/env python
# coding: utf-8
#fashionmnist_classification_convolution.ipynb
#특성맵 추출과정 - 컨볼루션 합성곱층과 풀링층을 거쳐 특성을 추출하는 방식
#  특성맵으로 추출된 특성을 완전연결층(flatten - danse) 에서 특성을 훈련한다.
# tf.keras.layers.Conv2D(
#     filters,            -  int, the dimension of the output space 
#     kernel_size,        - int or tuple/list of 2 integer, specifying the size of the convolution window. 
#     strides=(1, 1),     -int or tuple/list of 2 integer, specifying the stride length of the convolution
#     padding='valid',    -string, either "valid" or "same" (case-insensitive). "valid" means no padding.
#                               "same" results in padding evenly to the left/right or up/down of the input
#                                When padding="same" and strides=1, the output has the same size as the input. 
#     activation=None
# )
# tf.keras.layers.GlobalAveragePooling2D - 단일 특성맵  전체의 평균을 산출한다. (완전연결층을 보완)
# tf.keras.layers.averagePooling2D - 풀링 사이즈만큼 특성맵을 추출한다.
# tf.keras.layers.MaxPool2D(
#     pool_size=(2, 2),  - int or tuple of 2 integers, factors by which to downscale (dim1, dim2)
#     strides=None,      - int or tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
#     padding='valid',   - string, either "valid" or "same" (case-insensitive). "valid" means no padding. 
#                            "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. 
# )

import numpy as np
img_res = tf.keras.utils.load_img("examp.jfif",target_size=(52,52),interpolation='nearest',
                        keep_aspect_ratio=True)
img_res = np.array(img_res)
print(img_res.shape)
import matplotlib.pyplot as plt
plt.figure(figsize=(2,2))
plt.imshow(img_res)
plt.xticks([]);plt.yticks([])
plt.show()

img_res = img_res/255.
print(img_res[14])

#(52, 52, 3) -> (1, 52, 52, 3)
img_res = img_res.reshape(1,52,52,3)
print(img_res.shape)
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Conv2D, MaxPool2D,AveragePooling2D
test_model = Sequential()
test_model.add(Input((52,52,3)))
test_model.add(Conv2D(5,11,padding="same",activation="relu"))#필터사이즈 - 특성맵의 갯수
res = test_model(img_res)
print(res.shape)
plt.imshow(res[0,:,:,2])
plt.show()

test_model = (MaxPool2D(3,2))#풀사이즈와 스트라이드 증가시 윤곽만 또렷하고 이미지 사이즈는 축소된다.
res1 = test_model(img_res)
test_model =  MaxPool2D(5,3)#풀사이즈와 스트라이드 증가시 윤곽만 또렷하고 이미지 사이즈는 축소된다.
res2 = test_model(img_res)
print("res1:",res1.shape," res:2",res2.shape)
plt.figure(figsize=(5,5))
plt.subplot(2,3,1)
plt.imshow(res1[0,:,:,0])
plt.subplot(2,3,2)
plt.imshow(res1[0,:,:,1])
plt.subplot(2,3,3)
plt.imshow(res1[0,:,:,2])

plt.subplot(2,3,4)
plt.imshow(res2[0,:,:,0])
plt.subplot(2,3,5)
plt.imshow(res2[0,:,:,1])
plt.subplot(2,3,6)
plt.imshow(res2[0,:,:,2])
plt.show()
testavg = AveragePooling2D(5,3)
res3=testavg(img_res)
plt.subplot(1,4,1)
plt.imshow(res3[0,:,:,0])
plt.subplot(1,4,2)
plt.imshow(res3[0,:,:,1])
plt.subplot(1,4,3)
plt.imshow(res3[0,:,:,2])
plt.subplot(1,4,4)
plt.imshow(res3[0])
plt.show()