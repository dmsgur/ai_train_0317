#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Test_mnist_conv_advan
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
SPATH=r"checkpt/"
opt_model = tf.keras.models.load_model(f"{SPATH}select_6-0.04.keras");


# In[2]:


img3 = np.array(tf.keras.utils.load_img(r"test_img\3.jpg",color_mode='grayscale'))
img4 = np.array(tf.keras.utils.load_img(r"test_img\4.png",color_mode='grayscale'))
img5 = np.array(tf.keras.utils.load_img(r"test_img\6.jpg",color_mode='grayscale'))
img7 = np.array(tf.keras.utils.load_img(r"test_img\7.png",color_mode='grayscale'))
img8 = np.array(tf.keras.utils.load_img(r"test_img\8.png",color_mode='grayscale'))
img9 = np.array(tf.keras.utils.load_img(r"test_img\9.png",color_mode='grayscale'))
test_img = np.array([img3,img4,img5,img7,img8,img9])
print(test_img.shape)
test_img = (test_img.reshape(6,28,28,1)) #훈련데이터와 동일 모양 설정
test_img = 255-test_img #이미지 반전
print(test_img.shape)
plt.imshow(test_img[0],cmap="gray")
plt.show()


# In[3]:


y_pred = opt_model.predict(test_img)
print(y_pred.shape)


# In[4]:


plt.figure(figsize=(8,4))
for i,d in enumerate(test_img):
    plt.subplot(2,3,i+1)
    plt.imshow(d)
    plt.title(np.argmax(y_pred[i]))
plt.show()


# In[ ]:




