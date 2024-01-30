#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset=pd.read_csv('X_train_data.csv')


# In[3]:


dataset.head()


# In[4]:


X_train=dataset.drop(['Id','SalePrice'],axis=1)
y_train=dataset[['SalePrice']]


# In[6]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)


# In[7]:


feature_sel_model.get_support()


# In[8]:


selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))

# Assuming 'sel_' is defined somewhere in your code
# You need to define 'sel_' before accessing its attributes
# For example, if 'sel_' is a SelectFromModel object
print('features with coefficients shrank to zero: {}'.format(
    np.sum(feature_sel_model.estimator_.coef_ == 0)))


# In[9]:


selected_feat


# In[10]:


X_train=X_train[selected_feat]


# In[11]:


X_train.head()
X_train.to_csv('final_dataset.csv')


# In[18]:


get_ipython().system('pip install pycaret')
print("pycaret installed successfully!!")


# In[19]:


X_train.head()


# In[21]:


X_train.to_csv('X_train.csv', index=False)

