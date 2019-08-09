#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


churn_df=pd.read_csv('D:/pydata/churn.csv')
col_names=churn_df.columns.tolist()


# In[3]:


churn_df.head()


# In[4]:


col_names


# In[5]:


churn_result=churn_df['Churn?']
y=np.where(churn_result=='True.',1,0)


# In[6]:


to_drop=['State','Area Code','Phone','Churn?']


# In[7]:


churn_feat_space=churn_df.drop(to_drop,axis=1)


# In[8]:

#'yes'/'no'has to be converted to boolean values
yes_no_cols=["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols]=churn_feat_space[yes_no_cols]=='yes'


# In[9]:


features=churn_feat_space.columns
x=churn_feat_space.as_matrix().astype(np.float)


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)

print ("Feature space holds %d observations and %d features" % x.shape)
print ("Unique target labels:", np.unique(y))
print (x[0])
print (len(y[y == 0]))


# In[11]:


x


# In[12]:


from sklearn.model_selection import KFold


# In[13]:


def run_cv(x,y,clf_class,**kwargs):
    kf = KFold(n_splits=5,shuffle=True,random_state=1)
    y_pred = y.copy()

    for train_index, test_index in kf.split(y):
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(x_train,y_train)
        y_pred[test_index] = clf.predict(x_test)
    return y_pred


# In[14]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN


# In[15]:


def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)

print ("Support vector machines:")
print ("%.3f" % accuracy(y, run_cv(x,y,SVC)))
print('Random forest:')
print('%.3f' %accuracy(y,run_cv(x,y,RF)))
print('K-nearest-neighbors:')
print('%.3f' %accuracy(y,run_cv(x,y,KNN)))


# In[16]:


def run_prob_cv(x,y,clf_class,**kwargs):
    kf=KFold(n_splits=5,shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index,test_index in kf.split(y):
        x_train,x_test=x[train_index],x[test_index]
        y_train=y[train_index]
        clf=clf_class(**kwargs)
        clf.fit(x_train,y_train)
        y_prob[test_index]=clf.predict_proba(x_test)
    return y_prob


# In[17]:


pred_prob=run_prob_cv(x,y,RF,n_estimators=10)
pred_churn=pred_prob[:,1]
is_churn=y==1
counts=pd.value_counts(pred_churn)

true_prob={}
for prob in counts.index:
    true_prob[prob]=np.mean(is_churn[pred_churn==prob])
    true_prob=pd.Series(true_prob)
    
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns=['pred_prob','counts','true_prob']
counts


# In[ ]:




