#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import numpy as np
from sqlalchemy import create_engine
engine=create_engine('mysql+pymysql://root:Tsta@888@127.0.0.1:3306/data?charset=utf8')
import warnings
warnings.filterwarnings('ignore')
sql=pd.read_sql('all_gzdata',engine,chunksize=10000)


# In[ ]:


counts=[i['fullURLId'].value_counts() for i in sql]#逐块统计
counts=pd.concat(counts).groupby(level=0).sum()#合并统计结果，把相同的统计项合并
counts=counts.reset_index()#重新设置index，将原来的index作为counts的一列
counts.columns=['index','num']#重新设置列名，
counts['type']=counts['index'].str.extract('(\d{3})')#提取前3个数字作为类别id
counts_=counts[['type','num']].groupby('type').sum()#按类别合并
counts_.sort_values('num',ascending=False)#降序排列


# 点击与咨询相关(网页类型为101,"http://www.****.com/ask/")记录占了49.16%，其次是其他的类型(网页类型为199)占比24%左右，然后是知识相关(网页类型为107,"http://www.****.com/info/")占比22%左右。 用户点击的页面类型排行榜为：咨询相关、知识相关、其他方面的网页、法规（类型为301）、律师相关（类型为102）

# In[ ]:


#统计107类别的情况
def count107(i): #自定义统计函数
    j = [i['fullURLId'].str.contains('107')].copy() #找出类别包含107的网址
    j['type'] = None #添加空列
    j['type'][j['fullURL'].str.contains('info/.+?/')] = u'知识首页'
    j['type'][j['fullURL'].str.contains('info/.+?/.+?')] = u'知识列表页'
    j['type'][j['fullURL'].str.contains('/\d+?_*\d+?\.html')] = u'知识内容页'
    return j['type'].value_counts()
counts2 = [count107(i) for i in sql] #逐块统计
counts2 = pd.concat(counts2).groupby(level=0).sum() #合并统计结果


# In[ ]:


#统计点击次数
c = [i['realIP'].value_counts() for i in sql] #分块统计各个IP的出现次数
count3 = pd.concat(c).groupby(level = 0).sum() #合并统计结果，level=0表示按index分组
count3 = pd.DataFrame(count3) #Series转为DataFrame
count3[1] = 1 #添加一列，全为1
count3.groupby(0).sum() #统计各个“不同的点击次数”分别出现的次数


# In[ ]:


#数据清洗
for i in sql:
    d = i[['realIP', 'fullURL']] #只要网址列
    d = d[d['fullURL'].str.contains('\.html')].copy() #只要含有.html的网址
    #保存到数据库的cleaned_gzdata表中（如果表不存在则自动创建）
    d.to_sql('cleaned_gzdata', engine, index = False, if_exists = 'append')


# In[ ]:


#数据变换
for i in sql: #逐块变换并去重
    d = i.copy()
    d['fullURL'] = d['fullURL'].str.replace('_\d{0,2}.html', '.html') #将下划线后面部分去掉，规范为标准网址
    d = d.drop_duplicates() #删除重复记录
    d.to_sql('changed_gzdata', engine, index = False, if_exists = 'append') #保存


# In[ ]:


#网站分类
for i in sql: #逐块变换并去重
    d = i.copy()
    d['type_1'] = d['fullURL'] #复制一列
    d['type_1'][d['fullURL'].str.contains('(ask)|(askzt)')] = 'zixun' #将含有ask、askzt关键字的网址的类别一归为咨询（后面的规则就不详细列出来了，实际问题自己添加即可）
    d.to_sql('splited_gzdata', engine, index = False, if_exists = 'append') #保存


# In[ ]:


def Jaccard(a, b): #自定义杰卡德相似系数函数，仅对0-1矩阵有效
    return 1.0*(a*b).sum()/(a+b-a*b).sum()

class Recommender():
    sim=None #相似度矩阵
    def similarity(self, x, distance): #计算相似度矩阵的函数
        y = np.ones((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i,j] = distance(x[i], x[j])
        return y
    
    def fit (self, x, distance = Jaccard): #训练函数
        self.sim = self.similarity(x, distance)
        
    def recommend(self, a): #推荐函数
        return np.dot(self.sim, a)*(1-a)
  


# In[ ]:





# In[ ]:




