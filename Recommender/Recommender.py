#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
engine=create_engine('mysql+pymysql://root:Tsta@888@127.0.0.1:3306/data?charset=utf8')
import warnings
warnings.filterwarnings('ignore')
sql=pd.read_sql('all_gzdata',engine,chunksize=10000)


# In[ ]:


counts1=[i['fullURLId'].value_counts() for i in sql] # counts1此时是个列表
counts1=pd.concat(counts1).groupby(level_0).sum() # 由于sql分块的 上面counts1也是，concat纵向向链接
counts1=counts1.reset_index()
counts1.columns=['index','num']
counts1['type']=counts1['index'].str.extract('(\d{3})',expand=True)
counts1.head()
counts1_=counts1[['type','num']].groupby('type').sum()
counts1_.sort_values(by='num',ascending=False,inplace=True)
counts1_['percentage']=(counts1_['num']/counts1_['num'].sum())*100
counts1_.head()


# In[ ]:


a=counts1.set_index('type')
b=counts1.groupby('type').sum()
c=pd.merge(a,b,left_index=True,right_index=True)# 均以行索引为链接键 合并
c['percentage']=(c['num_x']/c['num_y'])*100
c.head()
c.reset_index(inplace=True) # inplace是否改变自身，True则改变原数据
d=c.sort_values(by=['type','percentage'],ascending=[True,False]).reset_index()
d.head()  


# In[ ]:


sql=pd.read_sql('all_gzdata',engine,chunksize=10000)
def count107(i):
    j=i[['fullURL']][i['fullURLId'].str.contains('107')].copy()
    j['type']=None
    j['type'][j['fullURL'].str.contains('info/.+?/')]='知识首页'
    j['type'][j['fullURL'].str.contains('info/.+?/.+?')]='知识列表页'
    j['type'][j['fullURL'].str.contains('/\d+?_*\d+?\.html')]='知识内容页'
    return j['type'].value_counts()
counts2=[count107(i) for i in sql]
counts2=pd.concat(counts2).groupby(level_0).sum()
counts2_=pd.DataFrame(counts2).reset_index()
counts2_['percentage']=(counts2_['type']/counts2_['type'].sum())*100
counts2_.columns=['107类型','num','百分比']
counts2.head()


# In[ ]:


def countquestion(i):
    j=i[['fullURLId']][i['fullURL'].str.contains('\?')].copy()
    return j
sql=pd.read_sql('all_gzdata',engine,chunksize=10000)
counts3=[countquestion(i)['fullURLId'].value_counts() for i in sql]
counts3=pd.concat(counts3).groupby(level_0).sum()
counts3
counts3_=pd.DataFrame(counts3)
counts3_['percentage']=(counts3_['fullURLId']/counts3_['fullURLId'].sum())*100
counts3_.reset_index(inplace=True)
counts3_.columns=['fullURLId','num','percentage']
counts3_.sort_values(by='percentage',ascending=False,inplace=True)
counts3_.reset_index(inplace=True)
c3=(counts3_['num'].sum()/counts1['num'].sum())*100
c3#统计?网页各类占比情况，探究?网页特征及在总样本占比


# In[ ]:


def page199(i):
    j=i[['pageTitle']][i['fullURLId'].str.contains('199') & i['fullURL'].str.contains('\?')]
    j['title']=1
    j['title'][j['pageTitle'].str.contains('法律快车-律师助手')]='法律快车-律师助手'
    j['title'][j['pageTitle'].str.contains('免费发布法律咨询')]='免费发布法律咨询'
    j['title'][j['pageTitle'].str.contains('咨询发布成功')]='咨询发布成功'
    j['title'][j['pageTitle'].str.contains('法律快搜')]='快搜'
    j['title'][j['title']==1]='其他类型'
    return j
sql=pd.read_sql('all_gzdata',engine,chunksize=10000)
counts4=[page199(i) for i in sql]
counts4_=pd.concat(counts4)
ct4_result=counts4_['title'].value_counts()
ct4_result=pd.DataFrame(ct4_result)
ct4_result.reset_index(inplace=True)
ct4_result['percentage']=(ct4_result['title']/ct4_result['title'].sum())*100
ct4_result.columns=['title','num','percentage']
ct4_result.sort_values(by='num',ascending=False,inplace=True)
ct4_result#统计199，其他类网址细分类别特征，占比


# In[ ]:


def wandering(i):
    j=i[['fullURLId','fullURL','pageTitle']][(i['fullURL'].str.contains('.html'))==False]
    return j
sql=pd.read_sql('all_gzdata',engine,chunksize=10000)
counts5=[wandering(i) for i in sql]
counts5_=pd.concat(counts5)

ct5Id=counts5_['fullURLId'].value_counts()
ct5Id
ct5Id_=pd.DataFrame(ct5Id)
ct5Id_.reset_index(inplace=True)
ct5Id_['index']=ct5Id_['index'].str.extract('(\d{3})',expand=True)
ct5Id_=pd.DataFrame(ct5Id_.groupby('index').sum())
ct5Id_['percentage']=(ct5Id_['fullURLId']/ct5Id_['fullURLId'].sum())*100
ct5Id_.reset_index(inplace=True)
ct5Id_.columns=['fullURLId','num','percentage']
ct5Id_.sort_values(by='num',ascending=False,inplace=True)
ct5Id_.reset_index(inplace=True)
#分析闲逛网站的用户浏览特征，哪些吸引了这些闲逛用户
ct5Id


# In[ ]:


sql=pd.read_sql('all_gzdata',engine,chunksize=10000)
counts6=[i['realIP'].value_counts() for i in sql]
counts6_=pd.concat(counts6).groupby(level=0).sum()
counts6_=pd.DataFrame(counts6_)
counts6_[1]=1
ct6=counts6_.groupby('realIP').sum()
ct6.columns=['用户数']    #  点击次数情况，来表征用户，即点击n次的用户数为m
ct6.index.name='点击次数' # realIP 出现次数 
ct6['用户百分比']=ct6['用户数']/ct6['用户数'].sum()*100
ct6['记录百分比']= ct6['用户数']*ct6.index/counts6_['realIP'].sum()*100   # 点击n次记录数占总记录数的比值
ct6.sort_index(inplace=True)
ct6_=ct6.iloc[:7,:].T
ct6['用户数'].sum()  #总点击次数
ct6_.insert(0,'总计',[ct6['用户数'].sum(),100,100])
ct6_['7次以上']=ct6_.iloc[:,0]-ct6_.iloc[:,1:].sum(axis=1)
ct6_


# In[ ]:


beyond7=ct6.index[7:]
bins=[7,100,1000,50000]  #用于划分区间
beyond7_cut=pd.cut(beyond7,bins,right=True,labels=['8~100','101~1000','1000以上'])#right=True 即(7,100],(101,1000]
beyond7_cut_=pd.DataFrame(beyond7_cut.value_counts())
beyond7_cut_.index.name='点击次数'
beyond7_cut_.columns=['用户数']
beyond7_cut_.iloc[0,0]=ct6.loc[8:100,:]['用户数'].sum()  #依次为点击在(7,100]用户数,iloc索引号比第几个少一
beyond7_cut_.iloc[1,0]=ct6.loc[101:1000,:]['用户数'].sum()  #注意，这里使用iloc会有问题，因这里行索引并非0开始的连续整数，而是名称索引，非自然整数索引
beyond7_cut_.iloc[2,0]=ct6.loc[1001:,:]['用户数'].sum()
beyond7_cut_.sort_values(by='用户数',ascending=False,inplace=True)
beyond7_cut_.reset_index(drop=True,inplace=True)
beyond7_cut_  # 点击8次以上情况统计分析，点击分布情况


# In[ ]:


for_once=counts6_[counts6_['realIP']==1]  # 分析点击一次用户行为特征
del for_once[1]  # 去除多余列
for_once.columns=['点击次数']
for_once.index.name='realIP'  # IP 找到，以下开始以此为基准 链接数据，merge

sql=pd.read_sql('all_gzdata',engine,chunksize=10000)
for_once_=[i[['fullURLId','fullURL','realIP']] for i in sql]
for_once_=pd.concat(for_once_)
for_once_merge=pd.merge(for_once,for_once_,right_on='realIP',left_index=True,how='left')
for_once_merge   #浏览一次用户行为信息，浏览的网页，可分析网页类型
for_once_url=pd.DataFrame(for_once_merge['fullURL'].value_counts())
for_once_url.reset_index(inplace=True)
for_once_url.columns=['fullURL','num']
for_once_url['percentage']=for_once_url['num']/for_once_url['num'].sum()*100
for_once_url  #浏览一次用户行为信息，浏览的网页，可分析网页类型:税法、婚姻、问题
# 分析用户点击情况，点击频率特征，点一次就走 跳失率。。


# In[ ]:


def url_click(i):
    j=i[['fullURL','fullURLId','realIP']][i['fullURL'].str.contains('\.html')] # \带不带都一样
    return j

sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
counts8=[url_click(i) for i in sql]
counts8_=pd.concat(counts8)
ct8=counts8_['fullURL'].value_counts()
ct8=pd.DataFrame(ct8)
ct8.columns=['点击次数']
ct8.index.name='网址'
ct8.sort_values(by='点击次数',ascending=False).iloc[:20,:]  #网址的点击率排行，用户关注度

ct8_=ct8.reset_index() #500和501一样结果，思路不一样，推第一个500
click_beyond_500=ct8_[(ct8_['网址'].str.contains('/\d+?_*\d+?\.html')) & (ct8_['点击次数']>50)]
click_beyond_501=ct8_[ct8_['网址'].str.contains('/\d+?_*\d+?\.html')][ct8_['点击次数']>50] # 会报警
#网页点击情况分析，点击最多是哪些，关注度
ct8


# In[ ]:


for i in sql: #逐块变换并去重
    d = i.copy()
    d['fullURL'] = d['fullURL'].str.replace('_\d{0,2}.html', '.html') #将下划线后面部分去掉，规范为标准网址
    d = d.drop_duplicates() #删除重复记录
    d.to_sql('changed_gzdata', engine, index = False, if_exists = 'append') #保存


# In[ ]:


def scanning_url(i):
    j=i.copy()
    j['fullURL']=j['fullURL'].str.replace('_\d{0,2}.html','.html')
    return j
sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
counts9=[scanning_url(i) for i in sql]
counts9_=pd.concat(counts9)
ct9=counts9_['fullURL'].value_counts()  # 每个网页被点击总次数，多页合为一页
ct9_=counts9_[['realIP','fullURL']].groupby('fullURL').count()  # 每个IP对所点击网页的点击次数，多页合为一页
ct9__=ct9_['realIP'].value_counts()
ct9__20=ct9__[:20]
ct9__20.plot()


# In[ ]:


#另一种分析视角，翻页行为统计分析
pattern=re.compile('http://(.*\d+?)_\w+_\w+\.html$|http://(.*\d+?)_\w+\.html$|http://(.*\w+?).html$',re.S)  # re.S 字符串跨行匹配;# 获取网址中以http://与.html中间的主体部分,即去掉翻页的内容，即去掉尾部"_d"
# 三个分别对应 主站点，
c9=click_beyond_500.set_index('网址').sort_index().copy()  #sort_index()保证了 同一主站翻页网页是按从第一页开始排列的，即首页、下一页，为下方计算下一页点击占上一页比率铺垫
c9['websitemain']=np.nan  # 统计主站点,即记录主站点，翻页的首页点击次数
for i in range(len(c9)):
    items=re.findall(pattern,c9.index[i])
    if len(items)==0:
        temp=np.nan
    else:
        for j in items[0]:
            if j!='':
                temp=j
    c9.iloc[i,1]=temp
c9.head()
c9_0=c9['websitemain'].value_counts()
c9_0=pd.DataFrame(c9_0)
c9_1=c9_0[c9_0['websitemain']>=2].copy() # 滤掉不存在翻页的网页
c9_1.columns=['Times']  # 用于识别是一类网页，主网址出现次数
c9_1.index.name='websitemain'
c9_1['num']=np.arange(1,len(c9_1)+1)
c9_2=pd.merge(c9,c9_1,left_on='websitemain',right_index=True,how='right')  # 链接左列与右索引 且左列与右索引的保留二者的行，且已右边为基础（即右边全保留，在此基础上添加左边的二者有共通行的）
# 当列与列做链接 索引会被忽略舍去，链接列与索引或索引与索引时索引会保留
c9_2.sort_index(inplace=True)

c9_2['per']=np.nan
def getper(x):
    print(x)
    for i in range(len(x)-1):
        x.iloc[i+1,-1]=x.iloc[i+1,0]/x.iloc[i,0]  # 翻页与上一页点击率比值，保存在最后一列处；同类网页下一页与上一页点击率比
    return x
result=pd.DataFrame()
for i in range(1,c9_2['num'].max()+1): #  多少种翻页类，c9_2['num'].max()+1，从1开始
    k=getper(c9_2[c9_2['num']==i])  # 同类翻页网页下一页与上一页点击数比值
    result=pd.concat([result,k])
c9_2['Times'].value_counts()

flipPageResult=result[result['Times']<10]
flipPageResult.head()
# 用户翻页行为分析，网站停留情况，文章分页优化


# In[ ]:


def countmidurl(i):  # 删除数据规则之中间网页（带midques_）
    j=i[['fullURL','fullURLId','realIP']].copy()
    j['type']='非中间类型网页'
    j['type'][j['fullURL'].str.contains('midques_')]='中间类型网页'
    return j['type'].value_counts()


# In[ ]:


sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
counts10=[countmidurl(i) for i in sql]
counts10_=pd.concat(counts10).groupby(level=0).sum()
counts10_


# In[ ]:


def count_no_html(i):  # 删除数据规则之目录页（带html）
    j=i[['fullURL','pageTitle','fullURLId']].copy()
    j['type']='有html'
    j['type'][j['fullURL'].str.contains('\.html')==False]="无html"
    return j['type'].value_counts()
sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
counts11=[count_no_html(i) for i in sql]
counts11_=pd.concat(counts11).groupby(level=0).sum()
counts11_


# In[ ]:


def count_law_consult(i):  # 数据删除规则之咨询、律师助手登录
    j=i[['fullURL','pageTitle','fullURLId']].copy()
    j['type']='其他'
    j['pageTitle'].fillna('空',inplace=True)
    j['type'][j['pageTitle'].str.contains('快车-律师助手')]='快车-律师助手'
    j['type'][j['pageTitle'].str.contains('咨询发布成功')]='咨询发布成功'
    j['type'][(j['pageTitle'].str.contains('免费发布法律咨询')) | (j['pageTitle'].str.contains('法律快搜'))]='快搜免费发布法律咨询'
    return j['type'].value_counts()
sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
counts12=[count_law_consult(i) for i in sql]
counts12_=pd.concat(counts12).groupby(level=0).sum()
counts12_


# In[ ]:


def count_law_ques(i):  # 数据删除规则之去掉与分析网站无关的网页
    j=i[['fullURL']].copy()
    j['fullURL']=j['fullURL'].str.replace('\?.*','')
    j['type']='主网址不包含关键字'
    j['type'][j['fullURL'].str.contains('lawtime')]='主网址包含关键字'
    return j
sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
counts13=[count_law_ques(i) for i in sql]
counts13_=pd.concat(counts13)['type'].value_counts()
counts13_


# In[ ]:


def count_duplicate(i):  # 数据删除规则之去掉同一用户同一时间同一网页的重复数据（可能是记录错误）
    j=i[['fullURL','realIP','timestamp_format']].copy()
    return j
sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
counts14=[count_duplicate(i) for i in sql]
counts14_=pd.concat(counts14)
print(len(counts14_[counts14_.duplicated()==True]),len(counts14_.drop_duplicates()))

ct14=counts14_.drop_duplicates()


# In[ ]:


# 开始对数据库数据进行清洗操作，构建模型数据
sql=pd.read_sql('select * from all_gzdata',engine,chunksize=10000)
for i in sql: #只要.html结尾的 & 截取问号左边的值 & 只要包含主网址（lawtime)的&网址中间没有midques_的
    d = i[['realIP', 'fullURL','pageTitle','userID','timestamp_format']].copy() # 只要网址列
    d['fullURL'] = d['fullURL'].str.replace('\?.*','') # 网址中问号后面的部分
    d = d[(d['fullURL'].str.contains('\.html')) & (d['fullURL'].str.contains('lawtime')) & (d['fullURL'].str.contains('midques_') == False)] # 只要含有.html的网址
    # 保存到数据库中
    d.to_sql('cleaned_one', engine, index = False, if_exists = 'append')


# In[ ]:


sql=pd.read_sql('select * from cleaned_one',engine,chunksize=10000)
for i in sql: #删除 快车-律师助手 & 免费发布法律咨询 & 咨询发布成功 & 法律快搜）
    d = i[['realIP','fullURL','pageTitle','userID','timestamp_format']]# 只要网址列
    d['pageTitle'].fillna(u'空',inplace=True)
    d = d[(d['pageTitle'].str.contains(u'快车-律师助手') == False) & (d['pageTitle'].str.contains(u'咨询发布成功') == False) & 
          (d['pageTitle'].str.contains(u'免费发布法律咨询') == False) & (d['pageTitle'].str.contains(u'法律快搜') == False)
         ].copy()
    # 保存到数据库中
    d.to_sql('cleaned_two', engine, index = False, if_exists = 'append')


# In[ ]:


sql=pd.read_sql('select * from cleaned_two',engine,chunksize=10000)
def dropduplicate(i): 
    j = i[['realIP','fullURL','pageTitle','userID','timestamp_format']].copy()
    return j


# In[ ]:


count15 = [dropduplicate(i) for i in sql]
count15 = pd.concat(count15)
print(len(count15)) # 2011695
count16 = count15.drop_duplicates(['fullURL','userID','timestamp_format']) # 一定要进行二次删除重复，因为不同的块中会有重复值
print(len(count16)) #　646915
count16.to_sql('cleaned_three', engine)
#数据清洗及保存到库的操作，清洗完毕


# In[ ]:


# 查看各表的长度
sql = pd.read_sql('all_gzdata', engine, chunksize = 10000)
lens = 0
for i in sql:
    temp = len(i)
    lens = temp + lens
print(lens)  #836877


# In[ ]:


# 查看cleaned_one表中的记录数
sql1 = pd.read_sql('cleaned_one', engine, chunksize = 10000)
lens1 = 0
for i in sql1:
    temp = len(i)
    lens1 = temp + lens1
print(lens1)#1341130


# In[ ]:


# 查看cleaned_two表中的记录数
sql2 = pd.read_sql('cleaned_two', engine, chunksize = 10000)
lens2 = 0
for i in sql2:
    temp = len(i)
    lens2 = temp + lens2
print(lens2)#2011695


# In[ ]:


# 查看cleaned_three表中的记录数
sql3 = pd.read_sql('cleaned_three', engine, chunksize = 10000)
lens3 = 0
for i in sql3:
    temp = len(i)
    lens3 = temp + lens3
print(lens3)


# In[ ]:


sql = pd.read_sql('cleaned_three', engine, chunksize = 10000)
l0 = 0
l1 = 0
l2 = 0
for i in sql:
    d = i.copy()
    # 获取所有记录的个数
    l0 += len(d)
    # 获取类似于http://www.lawtime.cn***/2007020619634_2.html格式的记录的个数
    # 匹配1 易知，匹配1一定包含匹配2
    x1 = d[d['fullURL'].str.contains('_\d{0,2}.html')]
    l1 += len(x1)
    # 匹配2
    # 获取类似于http://www.lawtime.cn***/29_1_p3.html格式的记录的个数
    x2 = d[d['fullURL'].str.contains('_\d{0,2}_\w{0,2}.html')]
    l2 += len(x2)
#    x1.to_sql('l1', engine, index=False, if_exists = 'append') # 保存
#    x2.to_sql('l2', engine, index=False, if_exists = 'append') # 保存
 
print(l0,l1,l2)


# In[ ]:


# 去掉翻页的网址
sql = pd.read_sql('cleaned_three', engine, chunksize = 10000)
l4 = 0
for i in sql:  #初筛
    d = i.copy()
    # 注意！！！替换1和替换2的顺序不能颠倒，否则删除不完整
    # 替换1 将类似于http://www.lawtime.cn***/29_1_p3.html下划线后面部分"_1_p3"去掉，规范为标准网址 
    d['fullURL'] = d['fullURL'].str.replace('_\d{0,2}_\w{0,2}.html','.html')#这部分网址有　9260　个
    # 替换2 将类似于http://www.lawtime.cn***/2007020619634_2.html下划线后面部分"_2"去掉，规范为标准网址
    d['fullURL'] = d['fullURL'].str.replace('_\d{0,2}.html','.html') #这部分网址有　55455-9260 = 46195 个
    d = d.drop_duplicates(['fullURL','userID']) # 删除重复记录(删除有相同网址和相同用户ID的)【不完整】因为不同的数据块中依然有重复数据
    l4 += len(d)
    d.to_sql('changed_1', engine, index=False, if_exists = 'append') # 保存
print(l4 )

# In[ ]:


sql = pd.read_sql('changed_1', engine, chunksize = 10000)
def dropduplicate(i):  #二次筛选
    j = i[['realIP','fullURL','pageTitle','userID','timestamp_format']].copy()
    return j
counts1 = [dropduplicate(i) for i in sql]
counts1 = pd.concat(counts1)
print(len(counts1))# 1095216

# In[ ]:


a = counts1.drop_duplicates(['fullURL','userID'])
print(len(a))# 528166
a.to_sql('changed_2', engine) # 保存


# In[ ]:


# 验证是否清洗干净，即changed_2已不存翻页网址
sql = pd.read_sql('changed_2', engine, chunksize = 10000)
l0 = 0
l1 = 0
l2 = 0
for i in sql:
    d = i.copy()
    # 获取所有记录的个数
    l0 += len(d)
    # 获取类似于http://www.lawtime.cn***/2007020619634_2.html格式的记录的个数
    # 匹配1 易知，匹配1一定包含匹配2
    x1 = d[d['fullURL'].str.contains('_\d{0,2}.html')]
    l1 += len(x1)
    # 匹配2
    # 获取类似于http://www.lawtime.cn***/29_1_p3.html格式的记录的个数
    x2 = d[d['fullURL'].str.contains('_\d{0,2}_\w{0,2}.html')]
    l2 += len(x2)
print(l0,l1,l2)# 528166 0 0表示已经删除成功


# In[ ]:


# 手动添加咨询类和知识类网址的类别，type={'咨询类','知识类'}
sql = pd.read_sql('changed_2', engine, chunksize = 10000)
def countzhishi(i):
    j = i[['fullURL']].copy()
    j['type'] = 'else'
    j['type'][j['fullURL'].str.contains('(info)|(faguizt)')] = 'zhishi'
    j['type'][j['fullURL'].str.contains('(ask)|(askzt)')] = 'zixun'
    return j
counts17 = [countzhishi(i) for i in sql]
counts17 = pd.concat(counts17)
counts17['type'].value_counts()
a = counts17['type'].value_counts()
b = pd.DataFrame(a)
b.columns = ['num']
b.index.name = 'type'
b['per'] = b['num']/b['num'].sum()*100
b


# In[ ]:


# 接上；咨询类较单一，知识类有丰富的二级、三级分类，以下作分析
c = counts17[counts17['type']=='zhishi']
d = c[c['fullURL'].str.contains('info')]
print(len(d)) # 102140
d['iszsk'] = 'else' # 结果显示是空  
d['iszsk'][d['fullURL'].str.contains('info')] = 'infoelsezsk' # 102032
d['iszsk'][d['fullURL'].str.contains('zhishiku')] = 'zsk' # 108
d['iszsk'].value_counts()  
# 由结果可知，除了‘info'和’zhishifku'没有其他类型，且 【info类型（不包含zhishiku)：infoelsezsk】和【包含zhishiku：zsk】类型无相交的部分。
# 因此分析知识类别下的二级类型时，需要分两部分考虑，求每部分的类别，再求并集，即为所有二级类型


# In[ ]:


# 对于http://www.lawtime.cn/info/jiaotong/jtsgcl/2011070996791.html类型的网址进行这样匹配,获取二级类别名称"jiaotong"
pattern = re.compile('/info/(.*?)/',re.S)
e = d[d['iszsk'] == 'infoelsezsk']
for i in range(len(e)): #用上面已经处理的'iszsk'列分成两种类别的网址，分别使用正则表达式进行匹配,较慢
    e.iloc[i,2] = re.findall(pattern, e.iloc[i,0])[0]
print(e.head())

# In[ ]:


# 对于http://www.lawtime.cn/zhishiku/laodong/info/***.html类型的网址进行这样匹配,获取二级类别名称"laodong"
# 由于还有一类是http://www.lawtime.cn/zhishiku/laodong/***.html，所以使用'zhishiku/(.*?)/'进行匹配
pattern1 = re.compile('zhishiku/(.*?)/',re.S)
f = d[d['iszsk'] == 'zsk']
for i in range(len(f)):
#     print i 
    f.iloc[i,2] = re.findall(pattern1, f.iloc[i,0])[0]
print(f.head())

e.columns = ['fullURL', 'type1', 'type2']
print(e.head())
f.columns = ['fullURL', 'type1', 'type2']
print(f.head())


# In[ ]:


# 将两类处理过二级类别的记录合并，求二级类别的交集
g = pd.concat([e,f])
h = g['type2'].value_counts()
# 求两类网址中的二级类别数，由结果可知，两类网址的二级类别的集合的并集满足所需条件
len(e['type2'].value_counts()) # 66
len(f['type2'].value_counts()) # 31
len(g['type2'].value_counts()) # 69
print(h.head())
print(h.index) # 列出知识类别下的所有的二级类别

for i in range(len(h.index)): # 将二级类别存入到数据库中
    x=g[g['type2']==h.index[i]]
    x.to_sql('h.index', engine, if_exists='append')


# In[ ]:


q = e.copy()
q['type3'] = np.nan
resultype3 = pd.DataFrame([],columns=q.columns)
for i in range(len(h.index)): # 正则匹配出三级类别
    pattern2 = re.compile('/info/'+h.index[i]+'/(.*?)/',re.S)
    current = q[q['type2'] == h.index[i]]
    print(current.head())
    for j in range(len(current)):
        findresult = re.findall(pattern2, current.iloc[j,0])
        if findresult == []: # 若匹配结果是空，则将空值进行赋值给三级类别
            current.iloc[j,3] = np.nan
        else:
            current.iloc[j,3] = findresult[0]
    resultype3 = pd.concat([resultype3,current])# 将处理后的数据拼接,即网址做索引，三列为一级、二级、三级类别
resultype3.set_index('fullURL',inplace=True)
resultype3.head(10)
# 统计婚姻类下面的三级类别的数目
j = resultype3[resultype3['type2'] == 'hunyin']['type3'].value_counts()
print(len(j)) # 145
j.head()


# In[ ]:


ype3nums = resultype3.pivot_table(index = ['type2','type3'], aggfunc = 'count') #类别3排序
# 方式2: Type3nums = resultype3.groupby([resultype3['type2'],resultype3['type3']]).count()
r = Type3nums.reset_index().sort_values(by=['type2','type1'],ascending=[True,False])
r.set_index(['type2','type3'],inplace = True)

r.to_excel('2_2_3Type3nums.xlsx')
r


# In[ ]:


# 读取数据库数据 ，属性规约，提取模型需要的数据（属性）；此处是只选择用户和用户访问的网页
sql = pd.read_sql('changed_2', engine, chunksize = 10000)
l1 = 0
l2 = 0 
for i in sql:
    zixun = i[['userID','fullURL']][i['fullURL'].str.contains('(ask)|(askzt)')].copy()
    l1 = len(zixun) + l1
    hunyin = i[['userID','fullURL']][i['fullURL'].str.contains('hunyin')].copy()    
    l2 = len(hunyin) + l2
    zixun.to_sql('zixunformodel', engine, index=False,if_exists = 'append')
    hunyin.to_sql('hunyinformodel', engine, index=False,if_exists = 'append')
print(l1,l2) # 393185 16982

# In[ ]:


# 方法二：
m = counts17[counts17['type'] == 'zixun']
n = counts17[counts17['fullURL'].str.contains('hunyin')]
p = m[m['fullURL'].str.contains('hunyin')]
p # 结果为空，可知，包含zixun的页面中不包含hunyin，两者没有交集
#savetosql(m,'zixun')
#savetosql(n,'hunyin')
m.to_sql('zixun',engine)
n.to_sql('hunyin',engine)


# In[ ]:


## 推荐，基于物品的协同过滤推荐、随机推荐、按流行度推荐

data=pd.read_sql('hunyinformodel',engine)
data.head()
def jaccard(a,b):  # 杰卡德相似系数，对0-1矩阵，故需要先将数据转成0-1矩阵
    return 1.0*(a*b).sum()/(a+b-a*b).sum()
class recommender():
    sim=None
    def similarity(self,x,distance):
        y=np.ones((len(x),len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i,j]=distance(x[i],x[j])
        return y
    def fit(self,x,distance=jaccard):
        self.sim=self.similarity(x,distance)
        return self.sim
    def recommend(self,a):
        return np.dot(self.sim,a)*(1-a)
len(data['fullURL'].value_counts())
len(data['realIP'].value_counts())
# 网址数远小于用户数，即商品数小于使用的客户数，采用物品的协同过滤推荐


# In[ ]:


start0=time.clock()
data.sort_values(by=['realIP','fullURL'],ascending=[True,True],inplace=True)
realIP=data['realIP'].value_counts().index
realIP=np.sort(realIP)
fullURL=data['fullURL'].value_counts().index
fullURl=np.sort(fullURL)
d=pd.DataFrame([],index=realIP,columns=fullURL)
for i in range(len(data)):
    a=data.iloc[i,0]
    b=data.iloc[i,1]
    d.loc[a,b]=1
d.fillna(0,inplace=True)
end0=time.clock()
usetime0=end0-start0
print('转化0-1矩阵耗时' + str(usetime0) +'s!')
#d.to_csv()


# In[ ]:


df=d.copy()
sampler=np.random.permutation(len(df))
df = df.take(sample)
train=df.iloc[:int(len(df)*0.9),:]  
test=df.iloc[int(len(df)*0.9):,:]  


# In[ ]:


df=df.values()
df_train=df[:int(len(df)*0.9),:]  # 9299
df_test=df[int(len(df)*0.9):,:]   # 1034
df_train=df_tain.T
df_test=df_test.T
print(df_train.shape,df_test.shape) # (4339L, 9299L) (4339L, 1034L)


# In[ ]:


start1=time.clock()
r=recommender()
sim=r.fit(df_train)  # 计算相似度矩阵
end1=time.clock()
a=pd.DataFrame(sim)
usetime1=end1-start1
print('建立矩阵耗时'+ str(usetime1)+'s!')
print(a.shape)


# In[ ]:


a.index=train.columns
#a.columns=train.columns
a.columns=train.index
a.head(20)


# In[ ]:


start2=time.clock()
result=r.recommend(df_test)
end2=time.clock()

result1=pd.DataFrame(result)
usetime2=end2-start2

print('测试集推荐函数耗时'+str(usetime2)+'s!')
result1.head()
result1.index=test.columns
result1.columns=test.index
result1


# In[ ]:


def xietong_result(k,recomMatrix): # k表推荐个数，recomMatrix推荐矩阵表DataFrame
    recomMatrix.fillna(0.0,inplace=True)
    n=range(1,k+1)
    recommends=['xietong'+str(y) for y in n]
    currentemp=pd.DataFrame([],index=recomMatrix.columns,columns=recommends)
    for i in range(len(recomMatrix.columns)):
        temp = recomMatrix.sort_values(by = recomMatrix.columns[i], ascending = False)
        j = 0 
        while j < k:
            currentemp.iloc[i,j] = temp.index[j]
            if temp.iloc[j,i] == 0.0:
                currentemp.iloc[i,j:k] = np.nan
                break
            j += 1
    return currentemp
start3 = time.clock()
xietong_result = xietong_result(3, result1)
end3 = time.clock()
print('按照协同过滤推荐方法为用户推荐3个未浏览过的网址耗时为' + str(end3 - start3)+'s!') #29.4996622053s!
xietong_result.head()


# In[ ]:


# test = df.iloc[int(len(df)*0.9):, :] #　所有测试数据df此时是矩阵，这样不能用
randata = 1 - df_test # df_test是用户浏览过的网页的矩阵形式，randata则表示是用户未浏览过的网页的矩阵形式
randmatrix = pd.DataFrame(randata, index = test.columns,columns=test.index)#这是用户未浏览过(待推荐）的网页的表格形式
def rand_recommd(K, recomMatrix):#　
    import random # 注意：这个random是random模块，
    import numpy as np
    
    recomMatrix.fillna(0.0,inplace=True) # 此处必须先填充空值
    recommends = ['recommed'+str(y) for y in range(1,K+1)]
    currentemp = pd.DataFrame([],index = recomMatrix.columns, columns = recommends)
    
    for i in range(len(recomMatrix.columns)): #len(res.columns)1
        curentcol = recomMatrix.columns[i]
        temp = recomMatrix[curentcol][recomMatrix[curentcol]!=0] # 未曾浏览过
    #     = temp.index[random.randint(0,len(temp))]
        if len(temp) == 0:
            currentemp.iloc[i,:] = np.nan
        elif len(temp) < K:
            r = temp.index.take(np.random.permutation(len(temp))) #注意：这个random是numpy模块的下属模块
            currentemp.iloc[i,:len(r)] = r
        else:
            r = random.sample(temp.index, K)
            currentemp.iloc[i,:] =  r
    return currentemp


# In[ ]:


start4 = time.clock()
random_result = rand_recommd(3, randmatrix) # 调用随机推荐函数
end4 = time.clock()
print('随机为用户推荐3个未浏览过的网址耗时为' + str(end4 - start4)+'s!') # 2.1900423292s!
#保存的表名命名格式为“3_1_k此表功能名称”，是本小节生成的第5张表格，功能为random_result：显示随机推荐的结果
#random_result.to_csv('random_result.csv')
random_result # 结果中出现了全空的行，这是冷启动现象,浏览该网页仅此IP一个，其他IP不曾浏览无相似系数


# In[ ]:


def popular_recommed(K, recomMatrix):
    recomMatrix.fillna(0.0,inplace=True)
    import numpy as np
    recommends = ['recommed'+str(y) for y in range(1,K+1)]
    currentemp = pd.DataFrame([],index = recomMatrix.columns, columns = recommends)
    
    for i in range(len(recomMatrix.columns)):
        curentcol = recomMatrix.columns[i]
        temp = recomMatrix[curentcol][recomMatrix[curentcol]!=0]
        if len(temp) == 0:
            currentemp.iloc[i,:] = np.nan
        elif len(temp) < K:
            r = temp.index #注意：这个random是numpy模块的下属模块
            currentemp.iloc[i,:len(r)] = r
        else:
            r = temp.index[:K]
            currentemp.iloc[i,:] =  r
            
    return currentemp  


# In[ ]:


# 确定用户未浏览的网页（可推荐的）的数据表格
TEST = 1-df_test  # df_test是用户浏览过的网页的矩阵形式，TEST则是未曾浏览过的
test2 = pd.DataFrame(TEST, index = test.columns, columns=test.index)
print(test2.head())
print(test2.shape )
 
# 确定网页浏览热度排名：
hotPopular = data['fullURL'].value_counts()
hotPopular = pd.DataFrame(hotPopular)
print(hotPopular.head())
print(hotPopular.shape)
 
# 按照流行度对可推荐的所有网址排序
test3 = test2.reset_index()
list_custom = list(hotPopular.index)
test3['index'] = test3['index'].astype('category')
test3['index'].cat.reorder_categories(list_custom, inplace=True)
test3.sort_values('index',inplace = True)
test3.set_index ('index', inplace = True)
print(test3.head())
print(test3.shape)
 
# 按照流行度为用户推荐3个未浏览过的网址
recomMatrix = test3  #
start5 = time.clock()
popular_result = popular_recommed(3, recomMatrix)
end5 = time.clock()
print('按照流行度为用户推荐3个未浏览过的网址耗时为' + str(end5 - start5)+'s!')#7.70043007471s!
 
#保存的表名命名格式为“3_1_k此表功能名称”，是本小节生成的第6张表格，功能为popular_result：显示按流行度推荐的结果
#popular_result.to_csv('3_1_6popular_result.csv')
 
popular_result


# In[ ]:





# In[ ]:




