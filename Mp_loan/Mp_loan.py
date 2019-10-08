#!/usr/bin/env python
# coding: utf-8
#数据下载：https://mirror.shileizcc.com/Other/LoanStats3a.csv
# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
loans_2007=pd.read_csv('./LoanStats3a.csv',skiprows=1)
#清除不必要的列
half_count=len(loans_2007)/2
loans_2007=loans_2007.dropna(thresh=half_count,axis=1)
loans_2007=loans_2007.drop(['desc','url'],axis=1)
loans_2007.to_csv('./loans_2007.csv',index=False)


# In[2]:


loans_2007=pd.read_csv('./loans_2007.csv')


# In[3]:


loans_2007.head()


# In[4]:


#清除无用数据
loans_2007=loans_2007.drop(['id','member_id','funded_amnt','funded_amnt_inv','grade','sub_grade','emp_title','issue_d'],axis=1)
loans_2007 = loans_2007.drop(['zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt','total_pymnt_inv', 'total_rec_prncp'], axis=1)
loans_2007 = loans_2007.drop(['total_rec_int', 'total_rec_late_fee','recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt'], axis=1)


# In[5]:


print(loans_2007.iloc[0])
print(loans_2007.shape[1])


# In[6]:


print(loans_2007['loan_status'].value_counts())


# In[7]:


loans_2007=loans_2007[(loans_2007['loan_status']=='Fully Paid') | (loans_2007['loan_status']=='Charged Off')]

status_replace={
    'loan_status':{
        'Fully Paid':1,
        'Charged Off':0,
    }
}
loans_2007=loans_2007.replace(status_replace)


# In[8]:


loans_2007.head()


# In[9]:


#去除空数据
orig_columns=loans_2007.columns
drop_columns=[]
for col in orig_columns:
    col_series=loans_2007[col].dropna().unique( )
    if len(col_series)==1:
        drop_columns.append(col)
loans_2007=loans_2007.drop(drop_columns,axis=1)


# In[10]:


loans_2007.to_csv('./filtered_loans_2007.csv',index=False)


# In[11]:


#统计缺失值
loans=pd.read_csv('./filtered_loans_2007.csv')
null_counts=loans.isnull().sum()


# In[12]:


null_counts


# In[13]:


#去掉缺失值较多的数据
loans=loans.drop('pub_rec_bankruptcies',axis=1)
loans=loans.dropna(axis=0)


# In[14]:


loans.dtypes.value_counts()


# In[15]:


#对object数据进行格式转换
mapping_dict={
    'emp_length':{
        '10+ years':10,
        '9 years':9,
        '8 years':8,
        '7 years':7,
        '6 years':6,
        '5 years':5,
        '4 years':4,
        '3 years':3,
        '2 years':2,
        '1 year':1,
        'n/a':0
    }
}
loans=loans.drop(['last_credit_pull_d','earliest_cr_line','addr_state','title'],axis=1)
loans['int_rate']=loans['int_rate'].str.rstrip('%').astype('float')
loans['revol_util']=loans['revol_util'].str.rstrip('%').astype('float')
loans=loans.replace(mapping_dict)


# In[16]:


cat_columns=['home_ownership','verification_status','emp_length','purpose','term']
dummy_df=pd.get_dummies(loans[cat_columns])
loans=pd.concat([loans,dummy_df],axis=1)
loans=loans.drop(cat_columns,axis=1)
loans=loans.drop('pymnt_plan',axis=1)

loans.to_csv('./cleaned_loans2007.csv',index=False)


# In[17]:


loans=pd.read_csv('./cleaned_loans2007.csv')


# In[18]:


loans.info()


# In[19]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
cols = loans.columns
train_cols = cols.drop('loan_status')
#得到特征
features = loans[train_cols]
#得到标签
target = loans['loan_status']
lr.fit(features, target)
predictions = lr.predict(features)


# In[20]:


#第一次测试——使用逻辑回归测试
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict,KFold
lr = LogisticRegression()
kf = KFold()
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

#定义指标
#True positives 
tp_filter = (predictions == 1) & (loans['loan_status']== 1) 
tp = len(predictions[tp_filter])
#False positives
fp_filter = (predictions == 1) & (loans['loan_status']== 0) 
fp = len(predictions[fp_filter])
#True negatives 
tn_filter = (predictions == 0) & (loans['loan_status']== 0) 
tn = len(predictions[tn_filter])
#False negatives 
fn_filter = (predictions == 0) & (loans['loan_status'] == 1) 
fn = len(predictions[fn_filter])
# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)


# In[22]:


#第二次测试——添加权重项，平衡数据的权重和
lr=LogisticRegression(class_weight='balanced')
kf=KFold()
predictions=cross_val_predict(lr,features,target,cv=kf)
predictions=pd.Series(predictions)

#True positives 
tp_filter = (predictions == 1) & (loans['loan_status']== 1) 
tp = len(predictions[tp_filter])
#False positives
fp_filter = (predictions == 1) & (loans['loan_status']== 0) 
fp = len(predictions[fp_filter])
#True negatives 
tn_filter = (predictions == 0) & (loans['loan_status']== 0) 
tn = len(predictions[tn_filter])
#False negatives 
fn_filter = (predictions == 0) & (loans['loan_status'] == 1) 
fn = len(predictions[fn_filter])
# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)


# In[23]:


#第三次测试——自定义权重项
penalty={
    0:5,
    1:1
}
lr=LogisticRegression(class_weight=penalty)
kf=KFold()
predictions=cross_val_predict(lr,features,target,cv=kf)
predictions=pd.Series(predictions)

#True positives 
tp_filter = (predictions == 1) & (loans['loan_status']== 1) 
tp = len(predictions[tp_filter])
#False positives
fp_filter = (predictions == 1) & (loans['loan_status']== 0) 
fp = len(predictions[fp_filter])
#True negatives 
tn_filter = (predictions == 0) & (loans['loan_status']== 0) 
tn = len(predictions[tn_filter])
#False negatives 
fn_filter = (predictions == 0) & (loans['loan_status'] == 1) 
fn = len(predictions[fn_filter])
# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)


# In[24]:


#第四次测试——随机森林
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,class_weight='balanced',random_state=1)
kf=KFold()
predictions=cross_val_predict(lr,features,target,cv=kf)
predictions=pd.Series(predictions)

#True positives 
tp_filter = (predictions == 1) & (loans['loan_status']== 1) 
tp = len(predictions[tp_filter])
#False positives
fp_filter = (predictions == 1) & (loans['loan_status']== 0) 
fp = len(predictions[fp_filter])
#True negatives 
tn_filter = (predictions == 0) & (loans['loan_status']== 0) 
tn = len(predictions[tn_filter])
#False negatives 
fn_filter = (predictions == 0) & (loans['loan_status'] == 1) 
fn = len(predictions[fn_filter])
# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)




