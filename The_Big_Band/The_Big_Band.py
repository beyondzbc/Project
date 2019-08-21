#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#配置乐队的夏天主题色
purple = (0.22,0.09,0.59) # 紫色
yellow = (0.99,0.89,0.27) # 黄色
green  = (0.36,0.94,0.55) # 绿色
blue   = (0.06,0.24,0.78) # 蓝色
red    = (0.98,0.31,0.36) # 红色


# In[3]:


data1=pd.read_excel('D:/pydata/乐队的夏天.xlsx','第一场')
data2=pd.read_excel('D:/pydata/乐队的夏天.xlsx','第二场两两PK赛')
data3=pd.read_excel('D:/pydata/乐队的夏天.xlsx','第三场累计积分赛')
data4=pd.read_excel('D:/pydata/乐队的夏天.xlsx','第四场复活赛')
data5=pd.read_excel('D:/pydata/乐队的夏天.xlsx','第五场9进7')
data6=pd.read_excel('D:/pydata/乐队的夏天.xlsx','第六场总决赛')


# In[4]:


data1.head()


# In[5]:


data1[data1['总得分'].isnull()]


# In[6]:


data1=data1.dropna(axis=0)


# In[7]:


data1


# In[8]:


data6.tail()


# In[9]:


fig,ax=plt.subplots(figsize=(12,12))
data1['总得分'].value_counts(bins=5,sort=False).plot.bar(ax=ax)

for tick in ax.get_xticklabels():
    tick.set_rotation(360)
    
ax.grid(False)


# In[10]:


#标准分函数定义
def z_score_normalize(series):
    mean=series.mean()
    std_dv=series.std()
    return series.apply(lambda x:(x-mean)/std_dv)


# In[11]:


#用循环的方式批量对每场比赛的得分做处理
competition=[data1,data2,data3,data4,data5,data6]

for period in competition:
    period['超级乐迷得分_标准分']=z_score_normalize(period['超级乐迷得分'])
    period['专业乐迷得分_标准分']=z_score_normalize(period['专业乐迷得分'])
    period['大众乐迷得分_标准分']=z_score_normalize(period['大众乐迷得分'])
    period['总得分_标准分']=z_score_normalize(period['总得分'])


# In[12]:


data1.sort_values(by='总得分_标准分',ascending=False)


# In[13]:


#每场数据是分开的，先截取原来的数据拼在一起成为一个长表
data1_score = data1[['场数','乐队','歌曲','排名','超级乐迷得分_标准分', '专业乐迷得分_标准分','大众乐迷得分_标准分','总得分_标准分']]
data2_score = data2[['场数','乐队','歌曲','排名','超级乐迷得分_标准分','专业乐迷得分_标准分','大众乐迷得分_标准分','总得分_标准分']]
data3_score = data3[['场数','乐队','歌曲','排名','超级乐迷得分_标准分','专业乐迷得分_标准分','大众乐迷得分_标准分','总得分_标准分']]
data4_score = data4[['场数','乐队','歌曲','排名','超级乐迷得分_标准分','专业乐迷得分_标准分','大众乐迷得分_标准分','总得分_标准分']]
data5_score = data5[['场数','乐队','歌曲','排名','超级乐迷得分_标准分','专业乐迷得分_标准分','大众乐迷得分_标准分','总得分_标准分']]
data6_score = data6[['场数','乐队','歌曲','排名','超级乐迷得分_标准分','专业乐迷得分_标准分','大众乐迷得分_标准分','总得分_标准分']]

total_score=pd.concat([data1_score,data2_score,data3_score,data4_score,data5_score,data6_score])


# In[14]:


total_score_mean=total_score.groupby(['乐队'])[['总得分_标准分']].mean().sort_values(by='总得分_标准分')

total_score_mean.tail(7).sort_values(by='总得分_标准分',ascending=False)


# In[15]:


#中文显示
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#对乐队总得分_标准分做可视化
y=np.arange(len(total_score_mean.index))
x=np.array(list(total_score_mean['总得分_标准分']))
fig,ax=plt.subplots(figsize=(12,12))
total_score_mean.plot.barh(ax=ax,alpha=0.7,title='27只乐队场均表现',color='g')
for a,b in zip(x,y):
    plt.text(a,b,'%.2f' % a,ha='center',va='center',fontsize=12)
ax.grid(False)


# In[16]:


total_score[total_score['乐队']=='痛仰乐队']
#再见杰克和公路之歌我经常单曲循环，传唱度比较高，看来我的审美还是符合主流


# In[17]:


total_score[total_score['乐队']=='面孔乐队']
#张三的歌我确实觉得不如原唱


# In[18]:


total_score[total_score['乐队']=='刺猬']
#火车一度单曲循环一整天


# In[44]:


total_score[total_score['乐队']=='旅行团乐队']
#周末玩具实在是不好听


# In[19]:


super_score_mean = total_score.groupby(['乐队'])[['超级乐迷得分_标准分']].mean().sort_values(by = '超级乐迷得分_标准分')
pro_score_mean = total_score.groupby(['乐队'])[['专业乐迷得分_标准分']].mean().sort_values(by = '专业乐迷得分_标准分')
public_score_mean = total_score.groupby(['乐队'])[['大众乐迷得分_标准分']].mean().sort_values(by = '大众乐迷得分_标准分')


# In[20]:


fig,ax=plt.subplots(1,3,figsize=(16,6))

super_score_mean.tail(5).plot.barh(ax=ax[0],color = '#dc2624',alpha=0.7,title='超级乐迷心中TOP5',grid=False)
pro_score_mean.tail(5).plot.barh(ax=ax[1],color = '#2b4750',alpha=0.7,title='专业乐迷心中TOP5',grid=False)
public_score_mean.tail(5).plot.barh(ax=ax[2],color = '#649E7D',alpha=0.7,title='大众乐迷心中TOP5',grid=False)


# In[21]:


super_score_mean_top5 = super_score_mean.tail(5)
pro_score_mean_top5 = pro_score_mean.tail(5)
public_score_mean_top5 = public_score_mean.tail(5)

super5=set(super_score_mean_top5.index)
pro5=set(pro_score_mean_top5.index)
pub5=set(public_score_mean_top5.index)

cross = super5&pro5&pub5
print('同时在三个群体中位列心目前五的乐队是:\n',cross)
print('\n')

super_pro = super5 - pro5
print('在超级乐迷心中前五，但不在专业乐迷心中的前五乐队是:\n',super_pro)
print('\n')

super_pub = super5 - pub5
print('在超级乐迷心中前五，但不在大众乐迷心中的前五乐队是:\n',super_pub)
print('\n')

pro_super = pro5 - super5
print('在专业乐迷心中前五，但不在超级乐迷心中的前五乐队是:\n',pro_super)
print('\n')

pro_pub = pro5 - pub5
print('在专业乐迷心中前五，但不在大众乐迷心中的前五乐队是:\n',pro_pub)
print('\n')

pub_super = pub5 - super5
print('在大众乐迷心中前五，但不在超级乐迷心中的前五乐队是:\n',pub_super)
print('\n')

pub_pro = pub5 - pro5
print('在大众乐迷心中前五，但不在专业乐迷心中的前五乐队是:\n',pub_pro)
print('\n')


# In[22]:


print('第五场中表演的九支乐队分别是：')
data5['乐队']


# In[ ]:


top9_rank = total_score[total_score['乐队'].isin(list(data5['乐队']))].iloc[:,[0,1,3]]

top9_rank = top9_rank.sort_values(by='场数')

fig = plt.figure(figsize=(16,10), dpi=100 )
ax = fig.add_subplot(1,1,1)

order = ['第一场','第二场','第二场复活赛','第三场上','第三场下','第四场','第五场','第六场上']

ax.plot(order,[0,0,0,0,0,0,0,0],c='w')

hgxs = top9_rank[top9_rank['乐队']=='海龟先生'].iloc[[0,3,1,2,4]]
ax.plot('场数','排名',data=hgxs,ls ='--',marker='o',markersize=30,label='海龟先生')

pnxl = top9_rank[top9_rank['乐队']=='盘尼西林'].iloc[[0,3,1,2,4,5]]
ax.plot('场数','排名',data=pnxl,ls ='--',marker='o',markersize=30,label='盘尼西林')

xkz = top9_rank[top9_rank['乐队']=='新裤子'].iloc[[0,3,1,2,4,5]]
ax.plot('场数','排名',data=xkz,ls ='--',marker='o',markersize=30,label='新裤子')

cw = top9_rank[top9_rank['乐队']=='刺猬'].iloc[[0,3,4,1,2,5,6]]
ax.plot('场数','排名',data=cw,ls ='--',marker='o',markersize=30,label='刺猬')

ty = top9_rank[top9_rank['乐队']=='痛仰乐队'].iloc[[0,1,5,3,2,4]]
ax.plot('场数','排名',data=ty,ls ='--',marker='o',markersize=30,label='痛仰乐队')

click = top9_rank[top9_rank['乐队']=='Click#15'].iloc[[0,1,2,6,3,4,5]]
ax.plot('场数','排名',data=click,ls ='--',marker='o',markersize=30,label='Click#15')

jlzr = top9_rank[top9_rank['乐队']=='九连真人'].iloc[[0,3,1,2,4,5]]
ax.plot('场数','排名',data=jlzr,ls ='--',marker='o',markersize=30,label='九连真人')

mk = top9_rank[top9_rank['乐队']=='面孔乐队'].iloc[[0,3,4,1,2,5]]
ax.plot('场数','排名',data=mk,ls ='--',marker='o',markersize=30,label='面孔乐队')

lxt = top9_rank[top9_rank['乐队']=='旅行团乐队'].iloc[[0,3,1,2,4,5]]
ax.plot('场数','排名',data=lxt,ls ='--',marker='o',markersize=30,label='旅行团乐队')

ax.set_xticks(np.arange(len(order)))
ax.set_yticks(np.arange(1,15))
ax.set_xticklabels(order)
ax.invert_yaxis()
ax.set_ylim(15,0)
ax.grid(False)
plt.title('九只队伍排名升降',fontdict = {'fontsize':20})
ax.legend(ncol=3)


#  歌曲分析

# In[ ]:


#得票率=得票数/总票数，然后变成一个0-1之间的数字


# In[24]:


#定义归一化函数并应用
def normalize(series,x_max):
    return series.apply(lambda x: x/x_max)


# In[25]:


for data in [data1,data2,data5,data6]:
    data['超级乐迷_归一分'] = normalize(data['超级乐迷得分'],50)

data3['超级乐迷_归一分'] = normalize(data3['超级乐迷得分'],60)
data4['超级乐迷_归一分'] = normalize(data4['超级乐迷得分'],40)

for data in [data1,data2,data3,data4,data5,data6]:
    data['专业乐迷_归一分'] = normalize(data['专业乐迷得分'],40)
    
for data in [data1,data2]:
    data['大众乐迷_归一分'] = normalize(data['大众乐迷得分'],100)
for data in [data3,data4,data5,data6]:
    data['大众乐迷_归一分'] = normalize(data['大众乐迷得分'],360)


# In[26]:


data6


# In[27]:


#批量对总得分应用归一化函数
match_total = {'data1':190,'data2':190,'data3':460,'data4':440,'data5':450,'data6':450}
data_list = [data1,data2,data3,data4,data5,data6]

for i,data in enumerate(data_list):
    data['总得分_归一分'] = normalize(data['总得分'],match_total['data'+ str(i+1)])


# In[28]:


col_index = [3,-4,-3,-2,-1]
total_nor_score = pd.DataFrame()
for data in data_list:
    total_nor_score = total_nor_score.append([data.iloc[:,col_index]])


# In[29]:


total_nor_score.sort_values(by = '总得分_归一分',ascending=False)


# In[30]:


total_nor_score.describe()


# In[31]:


fig,ax = plt.subplots(1,1,figsize = (8,4))
total_nor_score.boxplot(ax = ax,grid=False)
ax.set(title = '各组评委打分分布')


# In[32]:


top10_songs = total_nor_score.sort_values(by = '总得分_归一分').tail(10)
top10_songs.plot.barh(x='歌曲',y='总得分_归一分',color=purple)

print('截止第六期，最受欢迎的10首歌分别是:\n',list(top10_songs['歌曲']))


# In[33]:


total_nor_score.sort_values(by = '超级乐迷_归一分').tail(10).plot.barh(x='歌曲',y='超级乐迷_归一分',color=red)


# In[34]:


total_nor_score.sort_values(by = '专业乐迷_归一分').tail(10).plot.barh(x='歌曲',y='专业乐迷_归一分',color=yellow)


# In[35]:


total_nor_score.sort_values(by = '大众乐迷_归一分').tail(10).plot.barh(x='歌曲',y='大众乐迷_归一分',color=green)


# In[36]:


total_nor_score[total_nor_score['歌曲'] == '我愿意']


# In[37]:


total_nor_score_t = pd.merge(total_nor_score, total_score[['乐队','歌曲']], on='歌曲')
total_nor_score_t[total_nor_score_t['乐队'] == '海龟先生']


# In[38]:


total_nor_score_t[total_nor_score_t['乐队'] == '海龟先生'].mean()


# In[39]:


total_nor_score_t.iloc[[0,32,61,76]].mean()


# In[40]:


total_nor_score_t.groupby('乐队').mean().sort_values(by = '总得分_归一分',ascending = False).head(9)


# In[41]:


total_score_mean.sort_values(by = '总得分_标准分',ascending= False).head(9)


# In[42]:


# 这里面有些中途甚至很早就被淘汰了的乐队，如果把这些乐队排除，那么在最终的7只乐队里面，各组评委最喜欢谁呢？
import warnings
warnings.filterwarnings('ignore')
final_7 = ['新裤子','痛仰乐队','九连真人','Click#15','刺猬','盘尼西林','旅行团乐队']
final_7_super_score = super_score_mean.loc[final_7,].sort_values(by='超级乐迷得分_标准分')
final_7_pro_score = pro_score_mean.loc[final_7,].sort_values(by='专业乐迷得分_标准分')
final_7_public_score = public_score_mean.loc[final_7,].sort_values(by='大众乐迷得分_标准分')
final_7_total_score = total_score_mean.loc[final_7,].sort_values(by='总得分_标准分')

fig,ax = plt.subplots(4,1,figsize = (10,20))

final_7_total_score.plot.barh(ax=ax[0],color = yellow,alpha=0.7,title='我心中Hot5',grid=False)
final_7_super_score.plot.barh(ax=ax[1],color = '#dc2624',alpha=0.7,title='超级乐迷心中Hot5',grid=False)
final_7_pro_score.plot.barh(ax=ax[2],color = '#2b4750',alpha=0.7,title='专业乐迷心中Hot5',grid=False)
final_7_public_score.plot.barh(ax=ax[3],color = '#649E7D',alpha=0.7,title='大众乐迷心中Hot5',grid=False)


# In[43]:


data6.iloc[:,:-9]


# In[ ]:




