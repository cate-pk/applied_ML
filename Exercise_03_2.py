#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[49]:


from google.colab import files
uploaded = files.upload()
#BrainBody.txt 파일 불러오기
data = np.genfromtxt('BrainBody.txt', encoding='ascii')
#column 0 (index) 삭제
data = np.delete(data, np.s_[0:1], axis=1)
#column split
dataX, dataY = np.split(data, 2, axis = 1)


# In[ ]:


from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import optimizers


# In[ ]:


data = pd.read_fwf('BrainBody.txt',header=None, index_col=0)
data.columns = ["Brain", "Body"]


# In[52]:


data.head()


# In[53]:


data.shape


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Dense(10, input_shape = (1,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1))


# In[ ]:


sgd = optimizers.SGD(lr = 0.01)


# In[ ]:


model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])


# In[58]:


model.fit(data["Brain"], data["Body"], batch_size = 10, epochs = 20, verbose = 1)


# In[ ]:




