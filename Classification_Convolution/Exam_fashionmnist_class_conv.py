#!/usr/bin/env python
# coding: utf-8

# 1. 데이터 불러오기
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train: ",x_train.shape, " y_train: ",y_train.shape)
print("x_test: ",x_test.shape, " y_test: ",y_test.shape)
#라벨리스트 만들기
label_list = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 2. 데이터 구조 확인
print(x_train[1][14]) #정규화 필요
print(y_train[1]) #원핫인코딩 필요

# 3. 데이터 분할
from sklearn.model_selection import train_test_split
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=123, stratify=y_test)
#60%가 valid로, 40%가 test로
print(x_valid.shape)
print(x_test.shape)
print(y_valid.shape)
print(y_test.shape)

# 4. 데이터 셔플 및 전처리(정규화, 원핫인코딩)
import sklearn
x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=123)
#정규화 및 모양 변경 Conv2D는 최소 4차원으로 들어가야 함
#마지막에 -1(또는 1) 줘서 끝에 1차원이 추가됨
#gray이미지는 RGB 3차원이 아니라서 1차원을 추가해준 것
x_train = (x_train / 255.).reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], -1) 
x_test = (x_test / 255.).reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], -1)
x_valid = (x_valid / 255.).reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], -1)
y_train = tf.one_hot(y_train, len(label_list)) #원핫인코딩
y_valid = tf.one_hot(y_valid, len(label_list))
y_test = tf.one_hot(y_test, len(label_list))
print(x_train[0][14][:5])
print(x_valid[0][14][:5])
print(x_test[0][14][:5])
print(y_train[0])
print(y_valid[0])
print(y_test[0])

# 5. 정답과 이미지 일치여부 확인
import numpy as np
import matplotlib.pyplot as plt
t_rarr = np.random.randint(0, len(x_train), 5)
v_rarr = np.random.randint(0, len(x_valid), 5)
s_rarr = np.random.randint(0, len(x_test), 5)
print(t_rarr)
print(v_rarr)
print(s_rarr)

#zip 함수 이용
ix = 0
plt.figure(figsize=(5,5))
plt.rc("font", size=8)
plt.subplots_adjust(hspace=0.8)
for t,v,s in zip(t_rarr, v_rarr, s_rarr):
    plt.subplot(5, 3, ix+1)    
    plt.imshow(x_train[t], cmap="gray")
    plt.title(label_list[np.argmax(y_train[t])])
    plt.xticks([]); plt.yticks([]);    
    ix += 1
    plt.subplot(5, 3, ix+1)
    plt.imshow(x_valid[v], cmap="gray")
    plt.title(label_list[np.argmax(y_valid[v])])
    plt.xticks([]); plt.yticks([]);
    ix += 1
    plt.subplot(5, 3, ix+1)
    plt.imshow(x_test[s], cmap="gray")
    plt.title(label_list[np.argmax(y_test[s])])
    plt.xticks([]); plt.yticks([]);
    ix += 1
plt.show()    

# 6. 모델 구성
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

cmodel = Sequential()
cmodel.add(Input((28, 28, 1))) #이미지 사이즈 : 입력될 데이터의 모양
cmodel.add(Conv2D(10, 5, padding="same", activation="relu")) #컨볼루션(합성곱) 레이어 적용
cmodel.add(MaxPool2D(4, padding="same")) #풀링 적용
#흑백 이미지라 컨볼루션 레이어 2개, 컬러는 통상 3개 집어넣음
cmodel.add(Conv2D(20, 3, padding="same", activation="relu")) #컨볼루션(합성곱) 레이어 적용
cmodel.add(MaxPool2D(2, padding="same")) #풀링 적용
cmodel.add(Flatten()) #데잉터 벡터화
cmodel.add(Dropout(0.3)) #데이터 30% 제거
cmodel.add(Dense(128, activation="relu"))
cmodel.add(Dropout(0.3)) # 128 / 0.3 = 38.4
cmodel.add(Dense(32, activation="relu"))
cmodel.add(Dense(10, activation="softmax")) #최종 출력
#컴파일
cmodel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

# 7. 훈련 실행
#데이터가 많을 시 데이터 개수의 약배수만큼 batch_size를 넣어주는 게 좋다
#batch_size = 한 번에 모델에 입력되는 데이터의 개수
fhist = cmodel.fit(x_train,y_train,validation_data=(x_valid,y_valid),\
           epochs=100,batch_size=3000)

# 그래프 보기
print(fhist.history.keys())
plt.figure(figsize=(6,3))
plt.subplot(1,2,1) #정확도
plt.plot(fhist.history["acc"], label="train_acc")
plt.plot(fhist.history["val_acc"], label="valid_acc")
plt.legend()
plt.title("VALIDATION") 
plt.subplot(1,2,2) #손실율
plt.plot(fhist.history["loss"], label="train_loss")
plt.plot(fhist.history["val_loss"], label="valid_loss")
plt.legend()
plt.title("LOSSES") 
plt.show()

#훈련 30회
#정확도 그래프 : 훈련데이터보다 검증데이터 정확도가 더 높음 : 예측율이 좋을 것으로 예상됨
#손실율 그래프 : 손실율 감소 추세가 이어질 것으로 전망되므로, 훈련 횟수를 더 늘려도 될 것으로
#과대적합(검증 손실율이 훈련 손실율을 역전)하기 직전까지 훈련시키는 게 최선
#훈련 과정에서도 원하는 값을 지정하여 훈련을 중지시킬 수 있는 방법도 존재

#100회 결과 : loss: 0.2326 - acc: 0.9119 - val_loss: 0.2452 - val_acc: 0.9070

# 8. 모델 평가
# 모델평가는 훈련이 종료된 이후에 수행해야 함 :  훈련 전에는 가중치 값이 없기 때문.
# test 데이터ㅘ 라벨을 활용, 손실도와 정확율을 출력
lossVal, accVal = cmodel.evaluate(x_test, y_test)
print("손실도는 ", int(lossVal*10000)/10000, " 정확률은 ", int(accVal*10000)/100, "%")

# 9. 예측 및 예측값 시각화
y_pred = cmodel.predict(x_test)
rarr = np.random.randint(0, len(y_pred), 10) #랜덤정수 10개 생성, 인덱스로 활용
plt.figure(figsize=(5,5))
for ix, rix in enumerate(rarr):
    plt.subplot(2, 5, ix+1)
    plt.imshow(x_test[rix], cmap="gray")
    clr = "red"
    if np.argmax(y_test[rix]) == np.argmax(y_pred[rix]) : clr = "blue"
    plt.title(label_list[np.argmax(y_test[rix])], color=clr) #원핫인코딩된 y_test를 라벨로 전환
    plt.xlabel(label_list[np.argmax(y_pred[rix])], color=clr)    
    plt.xticks([]); plt.yticks([]);
plt.show()

# 10. 혼동행렬 : 예측 정답과 실제정답 일치화 - 레이블로 변경
# 혼동행렬은 양 데이터의 구조를 일치화시킨 후에 생성해야 한다.
print(y_test.shape)
print(y_pred.shape)
print(y_test[0]) #원핫인코딩 [0. 0. 0....]
print(y_pred[0]) #확률데이터 [1.5613245e-05 1.7114132-08]

#정수로 변경
y_real = np.argmax(y_test, axis=1) #행(axis=0)이 아닌, 실제 데이터가 들어있는 열(axis=1)을 기준으로
y_real_pred = np.argmax(y_pred, axis=1)
print(y_real[0])
print(y_real_pred[0])

#문자로 변경
y_real = [ label_list[d] for d in y_real ]
y_real_pred = [ label_list[d] for d in y_real_pred ]
print(y_real[:5])
print(y_real_pred[:5])

#labels 파라미터를 넣으면, 인덱스에 맞게 자동 반영
cm = sklearn.metrics.confusion_matrix(y_real, y_real_pred) #x축은 실제 정답, y축은 예측 정답
print(cm)

# 11. 혼동행렬의 시각화 : 히트맵(heatmap)
import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=label_list, yticklabels=label_list)
plt.show()
# Q. 아래와 같은 경우 가방, 셔츠 데이터만 학습시킬 시 다른 특성의 가중치에는 영향을 미치는지?

# 12. f1 score 구하기
#정밀도, 재현율, f1-score
print(sklearn.metrics.classification_report(y_real, y_real_pred))

# 13. 모델 저장
cmodel.save(r"./fashionmnist_convolution.keras")


