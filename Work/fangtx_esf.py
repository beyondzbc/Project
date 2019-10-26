#!/usr/bin/env python
# coding: utf-8

# In[731]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pyecharts.charts import Map,Pie,Bar
from wordcloud import WordCloud
import re
import seaborn as sns
from scipy import stats
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
plt.rc('figure',figsize=(20,20))
plt.rcParams['figure.dpi']=mpl.rcParams['axes.unicode_minus']=False
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[642]:


data=pd.read_csv('./project/fangtx.csv',encoding='gb18030')


# 数据观察

# In[643]:


data.head()
# data.shape
# data.sample(10)
# data.info()
# data.isnull().sum()


# In[644]:


data.describe()


# 数据预处理

# In[646]:


data['size'].unique()


# In[647]:


index=data[data['size'].isin(['联排','双拼','叠加','32室2厅','0室0厅','独栋'])].index
data=data.drop(index)#去除不合理数据和别墅、联排等类型


# In[648]:


data.shape


# In[649]:


data.isnull().sum()
data.dropna(axis=0,inplace=True)


# In[650]:


sns.boxplot(data=data['price_sum'])


# In[651]:


sns.boxplot(data=data['area'])


# In[652]:


data=data.drop(data[data['price']>200000].index)


# In[653]:


data=data.drop(data[data['area']>300].index)


# In[657]:


data['dire'].unique()
data['buildtime'].unique()
index=data[data['buildtime'].isin(['2021年建','未知','2020年建','2022年建','2024年建','2025年建'])].index
data=data.drop(index)#去除预计建成时间
data['floor'].unique()


# In[663]:


data.to_csv('./data_clean.csv',index=False,encoding='gb18030')


# In[678]:


data=pd.read_csv('./data_clean.csv',encoding='gb18030')
data.loc[data.city=='深圳',:].to_csv('./sz_data_clean.csv',index=False,encoding='gb18030')
data.loc[data.city=='广州',:].to_csv('./gz_data_clean.csv',index=False,encoding='gb18030')


# 数据分析与可视化

# In[679]:


#广东省房价总体情况
g=data.groupby('city')
r=g['price'].agg(['mean','count'])
r=r.sort_values('mean',ascending=False)
display(r)
r.plot(kind='bar')


# In[558]:


#广深两市房价远超其他城市，珠三角城市房价相对较高，粤东粤西边缘城市靠后


# In[605]:


import jieba
import jieba.analyse
rows=pd.read_csv('data_clean.csv', header=0,encoding='gb18030',dtype=str)
segments = []
for index, row in rows.iterrows():
    content = row[1]#提出小区名字的数据
    words = jieba.analyse.textrank(content, topK=50,withWeight=False,allowPOS=('ns', 'n', 'vn', 'v')) #TextRank 关键词抽取，只获取固定词性
    for word in words:# 记录全局分词
        segments.append({'word':word, 'count':1})        
dfSg = pd.DataFrame(segments)
# 词频统计
dfWord = dfSg.groupby('word')['count'].sum()
dfWord.sort_values(ascending=False)[:30]#取前30输出


# In[682]:


#房屋大小与面积关系
def area_price_relation_sz(city):
    data=pd.read_csv('./sz_data_clean.csv',encoding='gb18030')
    g=sns.jointplot(x='area',
                   y='price',
                   data=data, 
                   kind='reg' ,
                   stat_func=stats.pearsonr
                   )
    g.fig.set_dpi(100)
    g.ax_joint.set_xlabel('面积', fontweight='bold')
    g.ax_joint.set_ylabel('价格', fontweight='bold')
    return g

def area_price_relation_gz(city):
    data=pd.read_csv('./gz_data_clean.csv',encoding='gb18030')
    g=sns.jointplot(x='area',
                   y='price',
                   data=data, 
                   kind='reg' ,
                   stat_func=stats.pearsonr
                   )
    g.fig.set_dpi(100)
    g.ax_joint.set_xlabel('面积', fontweight='bold')
    g.ax_joint.set_ylabel('价格', fontweight='bold')
    return g


# In[684]:


area_price_relation_sz('深圳')
area_price_relation_gz('广州')


# In[787]:


# 地铁线分布与地铁线距离关系
def get_distance(city,data=data):
    station=[]#站
    distance=[]#距离
    station_count=[]#地铁线房源分布数量
    station_name=[]#地铁线
    data1=data.loc[data.city==city,:]
    data1=data1.reset_index(drop=True)#重置索引
    for i in range(len(data1)):
        s=re.findall('\d+',data1.loc[i,'advantage'])#用正则表达式匹配advantage标签
        if len(s)==2:
            distance.append(s[1])#距离
            station.append(s[0])#站线
            data1.loc[i,'distance']=s[1]
    data1.to_csv('{}_distance.csv'.format(city),index=False,encoding='gb18030')
#     data.to_csv('./gz_distance.csv',index=False,encoding='gb18030')#重新保存数据，后续进行分析
    count=list(set(station))#列表去掉重复值的方法
    count.sort()#列表排序
    for i in count:
        station_count.append( station.count('{}'.format(i)) )  #统计各个站线房源分布数量
        station_name.append('{}号线'.format(i))  #相应站线     
        
    plt.bar(station_name,station_count)
    plt.title('地铁房源分布')


# In[788]:


get_distance('深圳')


# In[767]:


get_distance('广州')


# In[770]:


def distance_price_relation(city_short):
    data=pd.read_csv('{}_distance.csv'.format(city_short),encoding='gb18030')
    g=sns.jointplot(x='distance',
                   y='price',
                   data=data.dropna(subset=['distance']),
                   kind='reg',
                    stat_func=stats.pearsonr
                   )
    g.fig.set_dpi(100)
    g.ax_joint.set_xlabel('最近地铁距离',fontweight='bold')
    g.ax_joint.set_ylabel('价格',fontweight='bold')
    return g


# In[771]:


distance_price_relation('深圳')


# In[772]:


distance_price_relation('广州')


# In[723]:


size_count=data['size'].value_counts().values[:5]
size_kind=data['size'].value_counts().index[:5]
# sns.barplot(size_kind,size_count)
plt.bar(size_kind,size_count)
plt.title('热门户型分布')


# In[732]:


time=data[data.city==city].buildtime.value_counts().index.tolist()[:5]
count=data[data.city==city].buildtime.value_counts().values.tolist()[:5]
labels=list(time)
plt.pie(count,labels=labels,autopct = '%3.1f%%',startangle = 180,shadow=True,colors=['c','r','gray','g','y'])
plt.title('建造年份统计')


# In[733]:


dire=data.dire.value_counts().index.tolist()
count=data.dire.value_counts().values.tolist()
labels=list(dire)
plt.pie(count,labels=labels,autopct = '%3.1f%%',startangle = 180,shadow=True,colors = ['c', 'r', 'y', 'g', 'gray'])
plt.title('房屋朝向')


# In[741]:


def distance_price_relation(city_short):
    data=pd.read_csv('./sz_distance.csv',encoding='gb18030')
    g=sns.jointplot(x='distance',
                   y='price',
                   data=data.dropna(subset=['distance']),
                   kind='reg',
                    stat_func=stats.pearsonr
                   )
    g.fig.set_dpi(100)
    g.ax_joint.set_xlabel('最近地铁距离',fontweight='bold')
    g.ax_joint.set_ylabel('价格',fontweight='bold')
    return g


# In[742]:


distance_price_relation('深圳')


# 机器学习预测房价

# In[833]:


sz_data=pd.read_csv('sz_distance.csv',encoding='gb18030')
gz_data=pd.read_csv('gz_distance.csv',encoding='gb18030')


# In[834]:


def transform(data):
    for i in range(len(data)):
        words=list(jieba.cut(data.loc[i,'advantage']))
        if '满二' in words:
            data.loc[i,'exemption of business tax']=1
        else:
            data.loc[i,'exemption of business tax']=0
        if '满五' in words:
            data.loc[i,'exemption of double tax']=1
        else:
            data.loc[i,'exemption of double tax']=0
        if '教育' in words:
            data.loc[i,'quality education']=1
        else:
            data.loc[i,'quality education']=0
            
transform(sz_data)
transform(gz_data)


# In[831]:


new_data=pd.DataFrame()
def datatrans(new_data,data,dire_sum=list(gz_data['dire'].unique())):
    new_data['city']=data['city']
    new_data['area']=data['area']
    new_data['buildtime']=data['buildtime']
    new_data['distance']=data['distance']
    for i in range(len(data)):
        s=re.findall('\d+',data.loc[i,'size'])
        new_data.loc[i,'room_num']=float(s[0])
        new_data.loc[i,'hall_num']=float(s[1])
        
        if '低层' in data.loc[i,'floor']:
            new_data.loc[i,'floor']=1
        elif '中层' in data.loc[i,'floor']:
            new_data.loc[i,'floor']=2
        elif '高层' in data.loc[i,'floor']:
            new_data.loc[i,'floor']=3
            
        dire=data.loc[i,'dire']
        idx=dire_sum.index(dire)+1
        new_data.loc[i,'dire']=idx
        
    new_data['exemption of business tax']=data['exemption of business tax']
    new_data['exemption of double tax']=data['exemption of double tax']
    new_data['quality education']=data['quality education']

datatrans(new_data,sz_data)
new_data1=pd.DataFrame()
datatrans(new_data1,gz_data)
new_data1=pd.concat([new_data1,new_data],axis=0,ignore_index=True)


# In[838]:


gz_price = gz_data['price']
sz_price = sz_data['price']
price = pd.concat([gz_price,sz_price],axis=0,ignore_index=True)
new_data1 = new_data1.join(pd.get_dummies(new_data1.city))
new_data1.drop('city',axis=1,inplace=True)
new_data1.to_csv('./new_data7.20.csv',index=False,encoding='gb18030')#房价预测，回归


# In[836]:


data=pd.read_csv('./new_data7.20.csv')
data.head()


# In[839]:


data=pd.read_csv('new_data7.20.csv')
data['distance'].fillna(5000,inplace=True)
data['buildtime'].fillna(data['buildtime'].mode()[0],inplace=True)
X = data
y=price

#数据分割，随机采样25%作为测试样本，其余作为训练样本
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

#数据标准化处理 归一化
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)


# In[840]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()     #初始化
lr.fit(x_train, y_train)    #训练数据
lr_y_predict = lr.predict(x_test)   #回归预测
#性能测评：使用R方得分指标对模型预测结果进行评价
from sklearn.metrics import  r2_score
print("LinearRegression模型的R方得分为：", r2_score(y_test, lr_y_predict))

plt.figure(figsize=(15, 5))
plt.plot(y_test.values[:100], "-r", label="真实值")
plt.plot(lr_y_predict[:100], "-g", label="预测值")
plt.legend()
plt.title("线性回归预测结果")


# In[847]:


param_grid = [
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,12)]
        
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,12)],
        'p':[i for i in range(1,6)]
    }
]
from sklearn.neighbors import KNeighborsRegressor
knnrgr = KNeighborsRegressor()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knnrgr,param_grid)
grid_search.fit(x_train,y_train)


# In[848]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[870]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet,SGDRegressor,BayesianRidge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
models = [Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),
GradientBoostingRegressor(),SVR(),ElasticNet(alpha=0.001,max_iter=10000),
SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),ExtraTreesRegressor(),
XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')]
names = [ "邻回归", "Lasso回归", "随机森林", "梯度提升树", "支持向量机" , "弹性网络","梯度下降回归","贝叶斯线性回归","极端随机森林回归","Xgboost回归"]
for name, model in zip(names, models):
    model.fit(x_train,y_train)
    predicted= model.predict(x_test)
    print("{}: {:.6f}, {:.4f}".format(name,model.score(x_test,y_test),mean_squared_error(y_test, predicted)))


# In[871]:


class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5,n_jobs=-1)
        grid_search.fit(X,y)
        print(grid_search.best_params_, grid_search.best_score_)
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])


# In[872]:


grid(Lasso()).grid_get(x_train,y_train,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})


# In[873]:


grid(Ridge()).grid_get(x_train,y_train,{'alpha':[35,40,45,50,55,60,65,70,80,90]})


# In[875]:


grid(ElasticNet()).grid_get(x_train,y_train,{'alpha':[0.0005,0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3,0.5,0.7],'max_iter':[10000]})


# In[ ]:




