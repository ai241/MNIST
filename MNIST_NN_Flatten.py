
# coding: utf-8

# In[1]:


from keras.datasets import mnist #impory mnist資料


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()#讀入mnist的train跟test


# In[3]:


import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt #畫圖用


# In[4]:


#看看資料長怎樣
x_train[0]


# In[5]:


#畫個圖看看
plt.imshow(x_train[0], cmap='gray_r')


# In[6]:


#建立模型
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical #把y的分類改成多維的向量


# In[7]:


#先把y的分類改成10維的向量
y_train_label = to_categorical(y_train)
y_test_label = to_categorical(y_test)


# In[8]:


y_train[0:3]


# In[9]:


y_train_label[0:3]


# In[10]:


#幾種不同模型的寫法
def get_model(mode):
    if mode == 1:
        #寫法1
        model = Sequential()
        model.add(Flatten(input_shape=(28,28))) #把dim=(28,28)的資料，拉成784個輸入
        model.add(Dense(input_dim=28*28, output_dim=500)) # 在新版keras中，output_dim已改為units
        model.add(Activation('sigmoid')) #也可改用relu之類的
        model.add(Dense(output_dim=500))
        model.add(Activation('sigmoid'))
        model.add(Dense(output_dim=10))
        model.add(Activation('softmax'))
    elif mode == 2:
        #寫法2
        model = Sequential()
        model.add(Flatten(input_shape=(28,28))) #把dim=(28,28)的資料，拉成784個input
        model.add(Dense(500,input_dim=28*28))
        model.add(Activation('sigmoid'))
        model.add(Dense(500))
        model.add(Activation('sigmoid'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
    else:
        #寫法3
        model = Sequential()
        model.add(Flatten(input_shape=(28,28))) #把dim=(28,28)的資料，拉成784個input
        model.add(Dense(500, activation='sigmoid', input_shape=(28*28,)))
        model.add(Dense(500, activation='sigmoid'))
        model.add(Dense(10, activation='softmax'))
    return model


# In[11]:


model = get_model(mode=1)
model.summary()


# In[12]:


#要如何衡量一個model的好壞，以及如何找最好的model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#僅分成兩類時，loss='binary_crossentropy'


# In[13]:


model.fit(x_train, y_train_label, batch_size=1000, nb_epoch=10)
#可直接把training set拆出一部分為validation set，如下：
#model.fit(x_train, y_train_label, batch_size=100, epochs=20, validation_spilt=0.25)
#validation_spilt=0.25：抽取25%的training data作驗證，這25%不在training中使用


# In[14]:


#看此model在test data上的準確率
score = model.evaluate(x_test,y_test_label)
print('Total loss on Testing Set', score[0])
print('Accuracy of Testing Set', score[1])


# In[15]:


#看此model在x_test上預測的值
result = model.predict(x_test)


# In[16]:


result[0:2] 


# In[17]:


result2 = [np.argmax(x) for x in result] #算一下模型給出的答案
np.array(result2[0:20])


# In[18]:


y_test[0:20]

