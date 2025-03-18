#!/usr/bin/env python
# coding: utf-8
#examp_LinearRegression_bostonHousing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Variables in order:
# CRIM 0    per capita crime rate by town (마을별 1인당 범죄율)
# ZN   1    proportion of residential land zoned for lots over 25,000 sq.ft.(주거용토지비율)
# INDUS 2   proportion of non-retail business acres per town(회사비율)
# CHAS  3   Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)(강가면 1 아니면 0)
# NOX   4   nitric oxides concentration (parts per 10 million)(공기질-일산화질소 농도)
# RM    5   average number of rooms per dwelling-평균 방수
# AGE   6   proportion of owner-occupied units built prior to 1940-주택 년한
# DIS   7   weighted distances to five Boston employment centres - 고용센터 5개 까지의 가중거리
# RAD   8   index of accessibility to radial highways-고속도로 접근성
# TAX   9   full-value property-tax rate per $10,000-재산세율
# PTRATIO  10 pupil-teacher ratio by town - 학생교사비율
# B     11   1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town-흑인비율
# LSTAT   12 % lower status of the population- 인구밀집도 낮은상태
# MEDV     Median value of owner-occupied homes in $1000's - 주택가격 (단위: 천달러)
print(x_train[0])#테이터파악

plt.figure(figsize=(7,6))
for ix in range(len(x_train[0])):
    plt.subplot(5,3,ix+1)
    plt.scatter(x_train[:,ix],y_train,s=3)
    plt.title(f"[{ix}]")
plt.show()    
# 인덱스 5번(평균방수)과 인덱스 12번(인구밀집도 낮은상태) 선형성 확인
# 데이터 분석 확인
print(x_train[0,5])
print(x_train[0,12])
print(" 평균 방수의 표준 편차 ",np.std(x_train[:,5]))
print(" 평균 방수의 최대방수 ",np.max(x_train[:,5]))
print(" 평균 방수의 최소방수 ",np.min(x_train[:,5]))
print(" 인구밀집낮은상태의 표준 편차 ",np.std(x_train[:,12]))
print(" 인구밀집낮은상태의 최대값 ",np.max(x_train[:,12]))
print(" 인구밀집낮은상태의 최소값 ",np.min(x_train[:,12]))
#결측값 , na, nan
print(sum(np.isnan(x_train[:,5])))
print(sum(np.isnan(x_train[:,12])))
# if np.isnan(np.NaN) :
#     print("참")
# else : print("거짓")

#히스토그램 - 데이터 분포와 이상값이 있는지 확인 가능
plt.hist(x_train[:,5])
plt.title("[5]")
plt.show()
plt.hist(x_train[:,12])
plt.title("[12]")
plt.show()
mean5=np.mean(x_train[:,5])
std5=np.std(x_train[:,5])
mean12=np.mean(x_train[:,12])
std12=np.std(x_train[:,12])
x_train[:,5]=(x_train[:,5]-mean5)/std5
x_train[:,12]=(x_train[:,12]-mean12)/std12
plt.figure(figsize=(7,2))
plt.subplot(1,2,1)
plt.hist(x_train[:,5])
plt.subplot(1,2,2)
plt.hist(x_train[:,12])
plt.show()
x_test[:,5]=(x_test[:,5]-mean5)/std5
x_test[:,12]=(x_test[:,12]-mean12)/std12
x_train = x_train[:,[5,12]]
x_test = x_test[:,[5,12]]
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#최종 데이터 값 확인
print(x_train[0])
print(x_test[0])
print(y_train[0])
print(y_test[0])
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Input(2,))
model.add(Dense(1))
model.compile(loss="MSE",optimizer="SGD")
fhist = model.fit(x_train,y_train,epochs=15)
print(fhist.history.keys())
plt.plot(fhist.history["loss"])
plt.show()
y_pred = model.predict(x_test)
print(y_pred.shape)
y_test = y_test.reshape(len(y_test),-1)
print(y_test.shape)
#전체 평균 정확률
y_acc = 1-(np.abs(y_test-y_pred)/y_test)
y_avg = np.mean(y_acc)*100
print(f"평균 정확률은 {y_avg:.2f} % ")
