"""
    가상화폐 일간 가격 분석
"""
import get_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from get_data import get_coindata
from get_data import create_rnndata#BTC_days_T30
from get_data import get_xpred
import pickle 
#1. 스케일 조정함수
def scalerX(x_data):
    x_scaler_mean = [] 
    x_scaler_std = []
    #정규분포 변환
    #z = (x - u) / s
    import numpy as np
    x_data = x_data.astype(np.float64)
    for i in range(len(x_data[0][0])):#각 필드 정규분포 변환
        xm = x_data[:,:,i].mean()
        xs = x_data[:,:,i].std()
        x_data[:,:,i] = (x_data[:,:,i]-xm)/xs
        x_scaler_mean.append(xm)
        x_scaler_std.append(xs)
    with open("BTC_days.pic","wb") as fp:
        pickle.dump({"x_scaler_mean":x_scaler_mean,"x_scaler_std":x_scaler_std},fp)
    return x_data

#2. 예측할 데이터 스케일 적용함수
def pred_scaler(x_data):
    with open("BTC_days.pic","rb") as fp:
        scdata = pickle.load(fp)
        x_scaler_mean = scdata["x_scaler_mean"]
        x_scaler_std = scdata["x_scaler_std"]
        print(x_scaler_mean)
        print(x_scaler_std)
    x_data = x_data.astype(np.float64)
    for i in range(len(x_data[0][0])):#각 필드 정규분포 변환
        x_data[:,:,i] = (x_data[:,:,i]-x_scaler_mean[i])/x_scaler_std[i]
    return x_data
#3. 데이터 수신 및 생성
rawdata = get_coindata("btc",to="2016-03-02 23:59:00")
x_data,y_data=create_rnndata(rawdata)
print(x_data.shape,y_data.shape)
x_data = scalerX(x_data)
print(x_data[0][0])
#4. 컨볼루션 LSTM 모델 구성
import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, ConvLSTM1D, Bidirectional,Reshape, GlobalAveragePooling1D,Dropout,GlobalMaxPool1D,AveragePooling1D,Flatten,MaxPooling1D
tf.random.set_seed(123)
np.random.seed(123)
model = Sequential()
model.add(Input((30,5)))# 30은 문장의 길이, 5 단어 임베딩
model.add(Reshape((30,5,1)))
conv_lstm1 = ConvLSTM1D(
    16,
    3,
    strides=1,
    padding='same',
    dropout=0.3,
    recurrent_dropout=0.5,
    return_sequences=True)
conv_lstm2 = ConvLSTM1D(
    32,
    5,
    strides=2,
    padding='same',
    dropout=0.4,
    recurrent_dropout=0.4,
    return_sequences=False)
model.add(Bidirectional(conv_lstm1))
model.add(Bidirectional(conv_lstm2))
# model.add(conv_lstm1)
# model.add(conv_lstm2)
model.add(AveragePooling1D())
model.add(Flatten())
model.add(Dropout(0.4,seed=123))
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.4,seed=123))
model.add(Dense(1,activation="linear"))
adam = tf.keras.optimizers.Adam(learning_rate=0.00009)
model.compile(loss="mse",optimizer=adam,metrics = ["mae"])

print(x_data.shape,y_data.shape)
#5. 모델 훈련 및 체크포인트 콜백함수 적용
filepath = "{epoch:02d}-{val_loss:.2f}.keras"
mck = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto',
    save_freq='epoch')
#callbacks=[mck],
y_data= y_data/100000000.
fhist = model.fit(x_data,y_data,validation_split=0.2,epochs=15,batch_size=len(x_data)//10)
#6. 모델 훈련결과 시각화
plt.subplot(1,2,1)
plt.plot(fhist.history["loss"],label="train mse")
plt.plot(fhist.history["val_loss"],label="valid mse")
plt.legend()
plt.title("MSE")
plt.subplot(1,2,2)
plt.plot(fhist.history["mae"],label="train mae")
plt.plot(fhist.history["val_mae"],label="valid mae")
plt.legend()
plt.title("MAE")
#7. 최적 모델 불러오기
model = tf.keras.models.load_model("BTC_days_T30.keras")
#8. 산점도 그래프 및 선형 그래프로 정답과 예측값 일치성 시각화
y_pred = model.predict(x_data)
plt .scatter(y_data,y_pred,label="pred _ acc",s=2,color="red")
plt .plot(y_data,y_data, label="true _ acc")
plt.show()
#BTC_days_T30

#9.예측할 오늘의 데이터와 어제의 데이터 수신
model = tf.keras.models.load_model("BTC_days_T30.keras")
x_today,x_yesday,y_curprice = get_xpred(coinname="BTC",getunit="days",timm="",timestep=30)
x_today=pred_scaler(np.array([x_today]))
x_yesday=pred_scaler(np.array([x_yesday]))
print(x_today.shape)
print(x_yesday.shape)
#10. 어제 데이터의 정확도 및 예측값의 오차율 출력과 오늘데이터의 예측값 출력
y_yes_pred = model.predict(x_yesday)
y_tod_pred = model.predict(x_today)
print("현재가격:",y_curprice,"현재예측가격:",y_yes_pred[0][0]*100000000, " 오차율:",(1-int(abs(y_yes_pred[0][0]*100000000/y_curprice)*100)/100)*100,"%")
print("예측가격:",y_tod_pred[0][0]*100000000)
