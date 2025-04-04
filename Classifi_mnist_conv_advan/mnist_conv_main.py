#!/usr/bin/env python
# coding: utf-8

# In[36]:


#mnist_conv_main
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
#시드값 고정
import random
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)


# In[37]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
print(x_train[0][14],y_train[0])
#셔플등의 기능들은 원핫인코딩 전에 완료하는게 좋다.
x_train,y_train = sklearn.utils.shuffle(x_train,y_train, random_state=123)
x_test,y_test = sklearn.utils.shuffle(x_test,y_test, random_state=123)


# In[38]:


#훈련데이터 전처리
# min-max-scaler
x_train=x_train/255.
y_train = tf.one_hot(y_train,10)
x_test=x_test/255.
y_test = tf.one_hot(y_test,10)
print(x_train[0][14],y_test[0])


# In[39]:


#conv 를 사용하기 위해 x 파일을 픽셀 단위변경
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print(x_train.shape,x_test.shape)


# In[40]:


import os
SPATH = r"checkpt/"
if not os.path.exists(SPATH):
    os.mkdir(SPATH)
filepath = SPATH+"{epoch}-{val_loss:.2f}.keras"
mcp = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)
espp = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    patience=10,
    verbose=1,
    restore_best_weights=True
)


# In[41]:


from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout


# In[42]:


model = Sequential()
model.add(Input((x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(Conv2D(10,3,padding="same",activation="relu"))
model.add(MaxPool2D(4,1))
model.add(Conv2D(30,3,padding="same",activation="relu"))
model.add(MaxPool2D(4,1))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10,activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])


# In[43]:


fhist = model.fit(x_train,y_train,validation_split=0.1,batch_size=30,epochs=200,
                 callbacks=[mcp,espp])


# In[52]:


model.save(f"{SPATH}val_acc99.keras")


# In[53]:


plt.subplot(1,2,1)
plt.plot(fhist.history["loss"],label="train_loss")
plt.plot(fhist.history["val_loss"],label="valid_loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(fhist.history["acc"],label="train_acc")
plt.plot(fhist.history["val_acc"],label="valid_acc")
plt.legend()
plt.show()


# In[55]:


#select_6-0.04.keras
optimal_model = tf.keras.models.load_model(f"{SPATH}select_6-0.04.keras");
res_opti = optimal_model.evaluate(x_test,y_test)
res = model.evaluate(x_test,y_test)
print(f"내가선택한 모델의 손실도:{res_opti[0]} 정확률:{res_opti[1]}")
print(f"조기종료가정확률로 선택한 모델의 손실도:{res[0]} 정확률:{res[1]}")


# In[68]:


rarr = np.random.randint(0,len(x_test),10)
print(rarr)


# In[69]:


y_pred = optimal_model.predict(x_test)
plt.figure(figsize=(8,3))
for i,d in enumerate(rarr):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i],cmap="gray")
    clr = "red" if np.argmax(y_test[i])!=np.argmax(y_pred[i]) else "blue"
    plt.title(np.argmax(y_test[i]),color=clr)
    plt.xlabel(np.argmax(y_pred[i]))
plt.show()
    


# In[ ]:




