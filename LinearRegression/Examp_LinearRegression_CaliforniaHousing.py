#!/usr/bin/env python
# coding: utf-8

import sklearn
datasets = sklearn.datasets.fetch_california_housing()
x_data = datasets["data"]
y_data = datasets["target"]
print(x_data.shape)
print(y_data.shape)
feature = datasets["feature_names"]
print(feature)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# MedInc- 평균 그룹의 중간 소득
# HouseAge- 평균 주택 년한
# AveRooms- 가구당 평균방의 갯수
# AveBedrms- 가구당 평균 침실수
# Population- 평균 인구수
# AveOccup- 평균 인원수
# Latitude- 위도
# Longitude- 경도
print(datasets["target_names"])#가격

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.subplot(3,3,1)
plt.scatter(x_data[:,0],y_data,s=1)
plt.title(feature[0])
plt.subplot(3,3,2)
plt.scatter(x_data[:,1],y_data,s=1)
plt.title(feature[1])
plt.subplot(3,3,3)
plt.scatter(x_data[:,2],y_data,s=1)
plt.title(feature[2])
plt.subplot(3,3,4)
plt.scatter(x_data[:,3],y_data,s=1)
plt.title(feature[3])
plt.subplot(3,3,5)
plt.scatter(x_data[:,4],y_data,s=1)
plt.title(feature[4])
plt.subplot(3,3,6)
plt.scatter(x_data[:,5],y_data,s=1)
plt.title(feature[5])
plt.subplot(3,3,7)
plt.scatter((x_data[:,6]+x_data[:,7])/len(x_data),y_data,s=1)
plt.title(feature[6])    
import pandas as pd
df = pd.DataFrame(x_data,columns=feature)
print(df.describe())
# 임계치 2(40) 3(5) 4(8000) 5(200)
plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
plt.hist(x_data[:,2])
plt.subplot(2,2,2)
plt.hist(x_data[:,3])
plt.subplot(2,2,3)
plt.hist(x_data[:,4])
plt.subplot(2,2,4)
plt.hist(x_data[:,5])
plt.show()
import numpy as np
#2(40) 3(5) 4(8000) 5(200)
def cutData(xdata,ydata):# 이상치 데이터 커팅
    tar2 = np.argwhere(xdata[:,2]>=40)
    xdata=np.delete(xdata,tar2,axis=0)
    ydata=np.delete(ydata,tar2,axis=0)

    tar3 = np.argwhere(xdata[:,3]>=5)
    xdata=np.delete(xdata,tar3,axis=0)
    ydata=np.delete(ydata,tar3,axis=0)   
    
    tar4 = np.argwhere(xdata[:,4]>=8000)
    xdata=np.delete(xdata,tar4,axis=0)
    ydata=np.delete(ydata,tar4,axis=0)
    
    tar5 = np.argwhere(xdata[:,5]>=200)
    xdata=np.delete(xdata,tar5,axis=0)
    ydata=np.delete(ydata,tar5,axis=0)    
    print(xdata.shape)
    print(ydata.shape)
    return xdata,ydata
x_data,y_data = cutData(x_data,y_data)
#2(40) 3(5) 4(8000) 5(200)
plt.subplot(2,2,1)
plt.hist(x_data[:,2])
plt.subplot(2,2,2)
plt.hist(x_data[:,3])
plt.subplot(2,2,3)
plt.hist(x_data[:,4])
plt.subplot(2,2,4)
plt.hist(x_data[:,5])
plt.show()
x_train,x_test,y_train,y_test = \
    sklearn.model_selection.train_test_split(x_data,y_data,test_size=0.2,random_state=111)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
for ix in range(8):
    mean1 = np.mean(x_train[:,ix])
    std1 = np.std(x_train[:,ix])
    x_train[:,ix] = (x_train[:,ix]-mean1)/std1
    x_test[:,ix] = (x_test[:,ix]-mean1)/std1
print(x_train[0])
print(x_test[0])
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Input((8,)))
model.add(Dense(1))
model.compile(loss="MAE",optimizer="SGD")
fhist = model.fit(x_data,y_data,epochs=100)

import matplotlib.pyplot as plt
plt.plot(fhist.history["loss"]
