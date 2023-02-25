#!/usr/bin/env python
# coding: utf-8

# # Diabetes

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_diabetes
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


diabetes = load_diabetes()
diabetes.keys()


# In[5]:


diabetes['DESCR']


# In[6]:


df = pd.DataFrame(diabetes['data'], columns=diabetes['feature_names'])
df.head()


# In[20]:


df_target = pd.DataFrame(diabetes['target'], columns=['target'])
df_target.head(3)


# In[7]:


df.info()


# In[8]:


df.describe().transpose()


# In[9]:


sns.heatmap(df.isnull(), cmap='plasma')


# In[94]:


# ? sum() ?
df.select_dtypes(include='object').columns


# In[93]:


sns.heatmap(df.corr(), cmap='plasma')


# In[16]:


sns.pairplot(df, hue='sex', palette='coolwarm')


# In[21]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(df.values, df_target.values, test_size=.3)


# In[96]:


import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import RootMeanSquaredError


# In[95]:


model = Sequential()
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=.5))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(rate=.5))
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(rate=.5))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss=MSE, optimizer='adam')


# In[103]:


log_dir = 'fit\log'
board = TensorBoard(log_dir, histogram_freq=1, write_graph=True, write_images=True, 
                    update_freq='epoch', profile_batch=2, embeddings_freq=1)


# In[104]:


early_callback = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=1)


# In[105]:


epochs = 600
model.fit(X_train, y_train, validation_data=(X_test, y_test), 
          epochs=epochs, batch_size=64, verbose=1, callbacks=[early_callback, board])


# In[106]:


history = pd.DataFrame(model.history.history)
history.head()


# In[107]:


history['loss'].plot()
history['val_loss'].plot()


# In[108]:


predictions = model.predict(X_test)


# In[109]:


sns.histplot((y_test - predictions), kde=True)


# In[111]:


print(log_dir)


# In[112]:


pwd


# In[ ]:




