#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(x_train,y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(x_train)


# In[4]:


len(x_test)


# In[5]:


len(y_train)


# In[6]:


len(y_test)


# In[7]:


x_train[0].shape


# In[8]:


x_train[0]


# In[9]:


plt.matshow(x_train[3])


# In[10]:


y_train[3]


# In[11]:


y_train[:5]


# In[12]:


x_train.shape


# In[18]:


x_train = x_train/255
x_test = x_test/255


# In[19]:


x_train[0]


# In[20]:


x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)
print(x_train_flatten.shape)
print(x_test_flatten.shape)


# In[21]:


x_test_flatten[0]


# In[22]:


x_train_flatten[0]


# In[23]:


### Creating a small neural network


# In[24]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,), activation='sigmoid')
])

model.compile(
       optimizer = 'adam',
       loss = 'sparse_categorical_crossentropy',
       metrics = ['accuracy']
)

model.fit(x_train_flatten, y_train, epochs = 100)


# In[25]:


model.evaluate(x_test_flatten, y_test)


# In[30]:


plt.matshow(x_test[2])


# In[33]:


y_predicted = model.predict(x_test_flatten)
y_predicted[2]


# In[34]:


np.argmax(y_predicted[2])


# In[35]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[36]:


y_test[:5]


# In[37]:


cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
cm


# In[39]:


import seaborn as sns
plt.figure(figsize = (10,8))
sns.heatmap(cm, annot= True, fmt = 'd')
plt.xlabel("Predicted")
plt.ylabel("Truth")


# In[42]:


model = keras.Sequential([
    keras.layers.Dense(100 , input_shape = (784,), activation='relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(
       optimizer = 'adam',
       loss = 'sparse_categorical_crossentropy',
       metrics = ['accuracy']
)

model.fit(x_train_flatten, y_train, epochs = 100)


# In[43]:


model.evaluate(x_test_flatten, y_test)


# In[44]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]
cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)

plt.figure(figsize = (10,8))
sns.heatmap(cm, annot= True, fmt = 'd')
plt.xlabel("Predicted")
plt.ylabel("Truth")


# In[46]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(100 ,activation='relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(
       optimizer = 'adam',
       loss = 'sparse_categorical_crossentropy',
       metrics = ['accuracy']
)

model.fit(x_train_flatten, y_train, epochs = 100)


# In[ ]:




