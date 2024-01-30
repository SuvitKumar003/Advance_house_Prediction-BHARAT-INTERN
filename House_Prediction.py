#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[93]:


dataset=pd.read_csv('house-prediction-data.csv')


# In[94]:


dataset.head()


# In[95]:


dataset.shape


# In[96]:


dataset.info()


# In[97]:


features_na=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1]
for feature in features_na:
    print(feature,'-> ',np.round(dataset[feature].isnull().sum()),'--> number of  missing values')


# In[98]:


for feature in features_na:
    print(feature,' -> ',np.round(dataset[feature].isnull().mean()*100,4),' --> % missing values ')


# In[99]:


data=dataset.copy()
for feature in features_na:
    data[feature]=np.where(data[feature].isnull(),1,0)


# In[100]:


print("1")


# In[101]:


for feature in features_na:
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
    # 1 represent the missing values and 0 represents the value is preset and we are calculating the relationship between missing values and saleprice


# In[102]:


data.head()


# In[103]:


#now we will get the list of variables with numerical variables
numerical_feature=[feature for feature in dataset.columns if dataset[feature].dtype != 'O']


# In[104]:


print(" total  number of features with numerical variable are ",len(numerical_feature))
for feature in numerical_feature:
    print(" the feature with numerical value is --> ",feature)


# In[105]:


dataset[numerical_feature].head()


# In[106]:


temporal_variables=[feature for feature in numerical_feature if 'Yr' in feature or 'Year' in feature]
print(temporal_variables)


# In[107]:


dataset.groupby('YrSold')['SalePrice'].plot()


# In[108]:


#as such no trend can be seen from the above diagram so we try plotting the above diagram with median of values
dataset.groupby('YrSold')['SalePrice'].mean().plot()


# In[109]:


dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Yr Sold')
plt.ylabel(' Median house price ')
plt.title(" house price vs yearsold")


# In[110]:


#since as we can see that the price is decreasing as the year increases but it should be reverse in trend so we will analyse it more
for feature in temporal_variables:
    if feature!='YrSold':
        data1=dataset.copy()
        
        data1[feature]=data1['YrSold']-data1[feature]
        plt.scatter(data1[feature],data1['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[111]:


discrete_variables = [feature for feature in numerical_feature if len(dataset[feature].unique()) < 25 and feature not in temporal_variables + ['Id']]
print("Discrete variables count:", len(discrete_variables))


# In[112]:


#Since it is discrete variable feature so will use bar plot to see the realtionship between the specific feature and the Sale price overhere.
for feature in discrete_variables:
    data = dataset.copy()
    sns.set_palette("Set1")  # Set the color palette
    ax = data.groupby(feature)['SalePrice'].median().plot.bar()
    ax.set_xlabel(feature)
    ax.set_ylabel('SalePrice')
    ax.set_title(feature)
    plt.show()


# In[113]:


continuous_variables = [feature for feature in numerical_feature if feature not in discrete_variables + ['Id']]
print(len(continuous_variables))


# In[114]:


for feature in continuous_variables:
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    #From the hostogram we can conclude that the data is skewed and need log transofmration


# In[115]:


for feature in continuous_variables:
    data=dataset.copy()
    plt.scatter(data[feature],data['SalePrice'])
   
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    #Scatter plot before doing log transformation on the datset


# In[116]:


for feature in continuous_variables:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# In[120]:


for feature in continuous_variables:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
    


# ###Since there are otuliers present in our dataset so willhandle them carefull so that they do not afftet our model

# In[121]:


# now we have analysed the continous , dsicrete varibales in our dataset now we will see the categorical feature
categorical_feature=[feature for feature in dataset.columns if data[feature].dtype=='O']


# In[122]:


print(" the number of categorical feature are ",len(categorical_feature))


# In[123]:


categorical_feature


# In[126]:


dataset[categorical_feature].head()


# now we need to convert the categorical features to dummy variables ini our to train our model

# In[128]:


for feature in categorical_feature:
    print(" the Feature is {} and number of categories are {} ".format(feature,len(dataset[feature].unique())))


# In[130]:


# finding the relationship between categorical feature and saleprice
for feature in categorical_feature:
    data = dataset.copy()
    sns.set_palette("Set2")  # Set the color palette
    ax = data.groupby(feature)['SalePrice'].median().plot.bar()
    ax.set_xlabel(feature)
    ax.set_ylabel('SalePrice')
    ax.set_title(feature)
    plt.show()


# In[ ]:



final_dataset=pd.read_csv('Final_datset.csv')
final_dataset.head()
final_dataset1=pd.concat([final_dataset,saleprice_target],axis=1)
final_dataset1.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X=final_dataset1.drop('SalePrice',axis=1)
y=final_dataset1['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.metrics import r2_score

from pycaret.regression import *
import pandas as pd
setup(data=final_dataset1, target='SalePrice', verbose=False)
cm=compare_models(fold=5)
gbr_model=create_model('gbr')
tuned_model=tune_model(gbr_model)
