#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset=pd.read_csv('house-prediction-data.csv')
dataset.head()


# In[4]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.3,random_state=42)


# In[5]:


X_train.shape, X_test.shape


# In[8]:


features_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtype == 'O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature, np.round(dataset[feature].isnull().mean() * 100, 2)))


# In[11]:


def replace_cat_feature(dataset,feature_nan):
    data=dataset.copy()
    data[feature_nan]=data[feature_nan].fillna('Missing')
    return data
dataset=replace_cat_feature(dataset,features_nan)
dataset[features_nan].isnull().sum()


# In[12]:


dataset.head()


# In[15]:


numerical_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtype!='O']
for feature in numerical_nan:
    print("{}: {}% missing values ".format(feature,np.round(dataset[feature].isnull().mean()*100,4)))


# In[18]:


for feature in numerical_nan:
    median_value=dataset[feature].median()
    
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)
dataset[numerical_nan].isnull().sum()


# In[19]:


dataset.head()


# In[27]:


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    
    dataset[feature]=dataset['YrSold']-dataset[feature]


# In[28]:


dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# In[29]:


dataset.head()


# In[31]:


num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[32]:


categorical_feature=[feature for feature in dataset.columns if dataset[feature].dtype=='O']


# In[38]:


print(categorical_feature)
for feature in categorical_feature:
    temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')


# In[37]:


dataset.head(100)


# In[41]:


for feature in categorical_feature:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


# In[43]:


dataset.head(10)


# In[45]:


scaling_feature=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]
print(len(scaling_feature))


# In[46]:


feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]
print(len(feature_scale))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[47]:


scaler.transform(dataset[feature_scale])


# In[48]:


# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                    axis=1)


# In[48]:





# In[49]:


data.head()


# In[52]:


data.to_csv('X_train_data.csv',index=False)

