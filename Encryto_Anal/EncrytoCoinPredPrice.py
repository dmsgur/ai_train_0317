#!/usr/bin/env python
# coding: utf-8

# In[30]:


#EncrytoCoinPredPrice(딥러닝 암호화폐 가격 분석 예측)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utilpy import getCandleData, creatX,integraion_xdata


# In[31]:


candle_data=getCandleData("days",cname="BTC")
x_datasets,y_datasets = creatX(candle_data,10)# 2param - 문제파일의 추출 갯수
x_datasets,featurelist = integraion_xdata(x_datasets)
print("기본값 일별 데이터 ")
print(x_datasets.shape)
print(y_datasets.shape)


# In[32]:


print(x_datasets[0])
print(featurelist)


# In[33]:


# opening_price = np.mean(x_datasets[:,:,0],axis=1)#(194, 5, 6)
# high_price = np.mean(x_datasets[:,:,1],axis=1)#(194, 5, 6)
# low_price = np.mean(x_datasets[:,:,2],axis=1)#(194, 5, 6)
# trade_price = np.mean(x_datasets[:,:,3],axis=1)#(194, 5, 6)
# candle_acc_trade_price = np.mean(x_datasets[:,:,4],axis=1)#(194, 5, 6)
# candle_acc_trade_volume = np.mean(x_datasets[:,:,5],axis=1)#(194, 5, 6)
# plt.figure(figsize=(8,3))
# plt.subplot(2,3,1)
# plt.scatter(opening_price,y_datasets[:,0],s=3)
# plt.subplot(2,3,2)
# plt.scatter(high_price,y_datasets[:,1],s=3)
# plt.subplot(2,3,3)
# plt.scatter(low_price,y_datasets[:,2],s=3)
# plt.subplot(2,3,4)
# plt.scatter(trade_price,y_datasets[:,3],s=3)
# # plt.subplot(2,3,5)
# # plt.scatter(candle_acc_trade_price,y_datasets,s=3)
# # plt.subplot(2,3,6)
# # plt.scatter(candle_acc_trade_volume,y_datasets,s=3)
# plt.show()
# print((opening_price[0]))
# print((opening_price[1]))
# print((opening_price[2]))
# print((opening_price[3]))
# print((opening_price[4]))
# print("===")
# for d in x_datasets:
#     sum=0
#     for a in d:
#         sum+=a[0]
#     sum/=5
#     print(sum)


# In[34]:


#산점도에 의해 연관성이 없이 마직막 두개의 데이터 삭제
x_datasets = np.delete(x_datasets,[-2,-1],axis=-1)
print(x_datasets[0])


# In[35]:


#데이터정규화
import sklearn
print(x_datasets.shape)
#z = (x - u) / s
m1 = x_datasets[:,:,0].mean()
s1 = x_datasets[:,:,0].std()
m2 = x_datasets[:,:,1].mean()
s2 = x_datasets[:,:,1].std()
m3 = x_datasets[:,:,2].mean()
s3 = x_datasets[:,:,2].std()
m4 = x_datasets[:,:,3].mean()
s4 = x_datasets[:,:,3].std()
x_datasets[:,:,0]=(x_datasets[:,:,0]-m1)/s1
x_datasets[:,:,1]=(x_datasets[:,:,1]-m2)/s2
x_datasets[:,:,2]=(x_datasets[:,:,2]-m3)/s3
x_datasets[:,:,3]=(x_datasets[:,:,3]-m4)/s4


# In[36]:


x_datasets = np.mean(x_datasets,axis=-1)
y_datasets = np.mean(y_datasets,axis=-1)
print(x_datasets.shape,y_datasets.shape)


# In[37]:


from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense,Dropout
model = Sequential()
model.add(Input((x_datasets.shape[1],)))
model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(1,activation="linear"))
layer_adam = tf.keras.optimizers.Adam(0.005)
model.compile(loss="mae",optimizer=layer_adam,metrics=["mse"])
print(x_datasets.shape);print(y_datasets.shape);


# In[38]:


fhist = model.fit(x_datasets,y_datasets,epochs=1000,batch_size=20)


# In[39]:


y_pred = model.predict(x_datasets)
print(y_pred.shape,y_datasets.shape)
y_pred = y_pred.reshape(y_pred.shape[0])
print(y_pred.shape)
print(y_pred[0])


# In[40]:


print(y_pred[0])
plt.plot(y_datasets,y_datasets,color="red")
plt.scatter(y_datasets,y_pred,s=3)
plt.show()


# In[41]:


print("오늘의 예측 가격정보:",end="")
print(f"최저:{y_pred[-2]*0.9:.2f} 최고:{y_pred[-2]*1.1:.2f} 평균:{y_pred[-2]:.2f}")
print("내일의 예측 가격정보:",end="")
print(f"최저:{y_pred[-1]*0.9:.2f} 최고:{y_pred[-1]*1.1:.2f} 평균:{y_pred[-1]:.2f}")
print("내일의 예측 상승하락율:",end="")
print(f"최저:{((y_pred[-1])/(y_pred[-2])-1)*100*0.9:.2f}%\
        최고:{((y_pred[-1])/(y_pred[-2])-1)*100*1.1:.2f}%\
        평균:{((y_pred[-1])/(y_pred[-2])-1)*100:.2f}%")


# In[42]:


plt.plot(y_datasets,label="True")
plt.plot(y_pred,label="Pred")
plt.legend()
plt.show()

