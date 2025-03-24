#!/usr/bin/env python
# coding: utf-8

# In[44]:


# 패션 영상 분류하기 : 데이터를 토대로 패션분류 10개로 분류
# 1. 필요 라이브러리 임포트
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense


# In[45]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
y_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Label	Description
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot


# In[46]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train[1000,14])
print(np.max(x_train))
print(np.min(x_train))
print(np.max(y_train)) #정답은 0~9까지 10가지 분류
# one hot encoding 원핫 인코딩 구조
# 0일시 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 2일시 [0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
# 9일시 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]...

#문제데이터 표준화 필요
#정답은 원핫인코딩이 필요
#sklearn train_test_split 분할은 원핫 인코딩 전에 수행해야 한다.


# In[47]:


# 2. 데이터 정규화 : min max
# 데이터 분할 : 검증데이터 50%, 테스트데이터 50%
# 기존에는 sklearn의 train_test_split 을 사용했으나
# 해당 모델에서는 Sequential의 validation_split 옵션을 활용한다.
x_train = x_train / 255 #RGB의 max 값은 255
x_test = x_test / 255
print(x_train[1000,14])


# In[48]:


# 직접 셔플링 해보기 : tf.random.shuffle 사용
# 셔플 : tensorflow, sklearn, numpy 등에서 사용가능
# x_train = tf.random.shuffle(x_train, seed=1111)
# y_train = tf.random.shuffle(y_train, seed=1111)

# sklearn의 shuffle 사용
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=1111)
x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)
print(y_train.shape)


# In[49]:


# shuffle가 잘 되어 있는지 테스트
np.random.seed(123)
rint = np.random.randint(0, len(x_train), 20) #0부터 x_train 개수만큼 20개
print(rint)
plt.figure(figsize=(8,8))
for ix in range(len(rint)):
    plt.subplot(5, 4, ix+1)    
    plt.imshow(x_train[rint[ix]], cmap="gray")
    plt.title(y_labels[y_train[rint[ix]]])
plt.show()


# In[50]:


# 3. 원 핫 인코딩
# 텐서플로우 - tf.one_hot(데이터, 구분 class 수량)
# 텐서플로우 - tf.keras.utils.to_categorical(데이터, num_classes = 구분 class 수량)
# print(y_train[5]) #도출되는 값의 인덱스가 1로 원핫인코딩됨
# print(y_train[6])
# print(y_train[7])
# print(tf.one_hot(y_train, 10)[5]) #원본 값이 4이므로 4번 인덱스 값이 1로 나머진 0으로 원핫 인코디됨
# print(tf.one_hot(y_train, 10)[6])
# print(tf.one_hot(y_train, 10)[7])

# 사이킷런 sklearn.preprocessing.OneHotEncoder() 객체로 만들어서 변형
import sklearn
# 원핫인코딩, 정수변경 모두 가능, 정답 레이블로 변환 가능
# sparse_output=False : 기본 반환타입인 희소행렬에서 Numpy 행렬 형태로 변경, 옵션 해제하고 반환값에 toarray() 함수적용해도 ok
# encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False) 
# test_onehot = encoder.fit_transform(y_train.reshape(len(y_train), -1), y_labels) #데이터와 라벨

# print(test_onehot[5])
# print(test_onehot[6])
# print(test_onehot[7])
# print(encoder.get_feature_names_out())


# In[51]:


# y_train = tf.one_hot(y_train, 10)
# y_test = tf.one_hot(y_test, 10)

# 직접 제작한 class로 원핫인코딩 해보기
from custom_encoder import CustomEncoder
encoder = CustomEncoder() #객체에 값 세팅됨
print(y_train.shape)
y_train = np.array(encoder.integer_to_one_hot(y_train, y_labels))
y_test = np.array(encoder.integer_to_one_hot(y_test, y_labels))
print(y_train.shape)
res = y_train[0]
print(y_train[0])
print(encoder.one_hot_to_label([res]))

# print(x_train.shape)
# print(x_test.shape)
# print(x_train[0][14])
# print(x_test[0][14])
# print(y_train.shape)
# print(y_test.shape)
# print(y_train[0])
# print(y_test[0])


# In[52]:


# 모델 구성 및 훈련
from tensorflow.keras.layers import Flatten
model = Sequential()
model.add(Input((28,28))) #이미지 데이터 사이즈
model.add(Flatten()) # 728*28 = 84개의 벡터로 변형 ( 완전 연결층 )
model.add(Dense(10, activation="softmax")) #종류 10개
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["acc"])
#validation_data : 훈련 데이터에서 검증데이터 20% 만큼 분리
#batch_size : 훈련 1회당 batch_size만큼 가져와 가중치를 수정한다?
fhist = model.fit(x_train, y_train, validation_split = 0.2, epochs=200, batch_size=5000)


# In[53]:


# 그래프 그리기
plt.subplot(1,2,1)
plt.plot(fhist.history["acc"], label="train_acc")
plt.plot(fhist.history["val_acc"], label="valid_acc")
plt.legend()
plt.subplot(1,2,2)
plt.plot(fhist.history["loss"], label="train_loss")
plt.plot(fhist.history["val_loss"], label="valid_loss")
plt.legend()
plt.show()


# In[54]:


# 모델평가 : 손신도 및 정확률 판단
res = model.evaluate(x_test, y_test)
print(res)
print("손실도는 ", int(res[0]*10000)/100, " 정확률은 ", int(res[1]*10000)/100)


# In[58]:


# 구성된 모델로 예측해보기
y_pred = model.predict(x_test)
print(y_test.shape)
print(y_pred.shape)
y_test_label = encoder.one_hot_to_label(y_test)
y_pred_label = encoder.one_hot_to_label(y_pred)
print(y_test_label[:5])
print(y_pred_label[:5])


# In[59]:


# 정답 레이블로 변경
# 실제 정답 레이블링
# 정수로 바꿨다는 표현 : 원핫 인코딩
# - 원래 정수 값을 원핫인코딩으로 변경, 해당 원핫인코딩을 다시 정수 값으로 변경
# one hot encoding된 값들 중 최대값은 1 하나 뿐이므로 index 추출 가능
def conv_label(c_data):
    #loop 돌리면서 원핫인코딩을 정답 값 인덱스 값으로 변경
    y_ix = np.array([ np.argmax(data) for data in c_data ])    
    #변경된 정수를 레이블 인덱스로 인출
    y_conv = np.array([y_labels[d] for d in y_ix])    
    return y_conv
y_test_conv = conv_label(y_test)  #실제 정답 값
y_pred_conv = conv_label(y_pred)  #모델 예측 값

print(y_test_conv[:10])
print(y_pred_conv[:10])


# In[60]:


# custom_encoder.py 테스트
np.random.seed(123)
rarr = np.random.randint(0, len(y_test_label), 10)
print(rarr)


# In[63]:


plt.subplots_adjust(wspace=1, hspace=0.001)
for ix, data in enumerate(rarr): #enumerate 인덱스와 값을 함께 출력
    plt.subplot(2,5,ix+1)
    plt.imshow(x_test[data], cmap="gray")
    clr = y_test_label[data] == y_pred_label[data] #정답과 예측값
    plt.title("True: "+y_test_label[data])
    plt.xlabel("Pred: "+y_pred_label[data], color=("blue" if clr else "red" ))
    plt.xticks([]); plt.yticks([]); #간격 조절
plt.show()


# In[65]:


#모델 저장하기
tf.keras.models.save_model(model, r"./save_model/fashion_mnist_classification.keras") #model.save("model.keras")


# In[66]:


import pickle #객체저장 내장 라이브러리, 현재 디렉토리 하위 save_model 폴더 내에 저장 mode=wb(write binary)
with open(r"./save_model/fashion_mnist.classification_encoder", "wb") as fp: 
    pickle.dump(encoder, fp)

