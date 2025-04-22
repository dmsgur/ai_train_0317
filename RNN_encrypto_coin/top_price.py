
#최고가 구하기
import get_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from get_data import get_coindata
from get_data import high_create_rnndata#BTC_days_T30
from get_data import get_xpred,get_highxpred
import pickle 
pre_fix="high_"

def scalerX(x_data):
    print(x_data.shape)
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
    with open(pre_fix+"BTC_days.pic","wb") as fp:
        pickle.dump({pre_fix+"x_scaler_mean":x_scaler_mean,pre_fix+"x_scaler_std":x_scaler_std},fp)
    return x_data
def pred_scaler(x_data):
    print("===============",x_data.shape)
    with open(pre_fix+"BTC_days.pic","rb") as fp:
        scdata = pickle.load(fp)
        x_scaler_mean = scdata[pre_fix+"x_scaler_mean"]
        x_scaler_std = scdata[pre_fix+"x_scaler_std"]
        print(x_scaler_mean)
        print(x_scaler_std)
    x_data = x_data.astype(np.float64)
    for i in range(len(x_data[0][0])):#각 필드 정규분포 변환
        x_data[:,:,i] = (x_data[:,:,i]-x_scaler_mean[i])/x_scaler_std[i]
    return x_data

#데이터 수신 및 생성
rawdata = get_coindata("btc",to="2016-03-02 23:59:00")
x_data,y_data=high_create_rnndata(rawdata)
print(x_data.shape,y_data.shape)
x_data = scalerX(x_data)
print(x_data[0][0])

import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, ConvLSTM1D, Bidirectional,Reshape, GlobalAveragePooling1D,Dropout,GlobalMaxPool1D,AveragePooling1D,Flatten,MaxPooling1D
tf.random.set_seed(123)
np.random.seed(123)
model = Sequential()
model.add(Input((30,1)))# 30은 문장의 길이, 5 단어 임베딩
ls1 = LSTM(
    32,
   dropout=0.3,
   recurrent_dropout=0.3,
    return_sequences=True)
model.add(ls1)
ls2 = LSTM(
    32,
    dropout=0.3,
    recurrent_dropout=0.3,
    return_sequences=True)
model.add(ls2)
ls3 = LSTM(
    64,
    dropout=0.3,
    recurrent_dropout=0.3,
    return_sequences=False)
model.add(ls3)
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4,seed=123))
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="linear"))
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="mse",optimizer=adam,metrics = ["mae"])

print(x_data.shape,y_data.shape)

filepath = pre_fix+"{epoch:02d}-{val_loss:.2f}.keras"
mck = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto',
    save_freq='epoch')
#callbacks=[mck],
y_data= y_data/100000000.
fhist = model.fit(x_data,y_data,validation_split=0.2,epochs=50,callbacks=[mck],batch_size=len(x_data)//10)

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

#예측값 
y_pred = model.predict(x_data)
plt .scatter(y_data,y_pred,label="pred _ acc",s=2,color="red")
plt .plot(y_data,y_data, label="true _ acc")
plt.show()

#BTC_days_T30
model = tf.keras.models.load_model(pre_fix+"BTC_days.keras")
x_today,x_yesday,y_curprice = get_highxpred(coinname="BTC",getunit="days",timm="",timestep=30)
x_today = x_today.reshape(1,30,1)
x_yesday = x_yesday.reshape(1,30,1)
x_today=pred_scaler(x_today)
x_yesday=pred_scaler(x_yesday)
print(x_today.shape)
print(x_yesday.shape)

y_yes_pred = model.predict(x_yesday)
y_tod_pred = model.predict(x_today)
print("현재최고가격:",y_curprice,"현재예측최고가격:",y_yes_pred[0][0]*100000000, " 오차율:",(1-int(abs(y_yes_pred[0][0]*100000000/y_curprice)*100)/100)*100,"%")
print("예측최고가격:",y_tod_pred[0][0]*100000000)