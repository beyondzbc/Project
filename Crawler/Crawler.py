#!/usr/bin/env python
# coding: utf-8

# requests是python的一个第三方库，比自带的urllib更加简单方便和人性化  
# 使用requests获取网页的源代码，最简单的情况下只需要两行代码  
# Get与Post方法使用 对于大多数直接在浏览器输入网址就能访问的网页，使用requests的GET方法就能获取到网页的源代码  
# 这里需要使用 .content 来显示网页的源代码。如果不加 .content, 则得到的只会是网页访问的状态码，类似于下面这样： <Response [200]>

# In[3]:


import requests
html = requests.get('https://shileizcc.com').content


# 对于不能直接在浏览器中输入网址访问的页面，就需要使用requests的POST方法来获取源代码  
# data 这个字典的内容和项数需要根据实际情况做修改

# In[4]:


import requests
data = {'key1': 'value1',
        'key2': 'value2'}
html = requests.post('网址', data=data).content


# 还有一些网址，提交的内容需要是 json 格式，因此我们的 post 方法的参数可以做一些修改：

# In[5]:


html = requests.post('https://shileizcc.com', json=data).content


# xpath是一种HTML和XML查询语言，能在HTML和XML的树状结构中寻找节点，在Python中安装lxml库来使用xpath技术  
# 使用xpath 语句：/html/body/div[@class="useful"]/ul/li/text()

# In[6]:


source='''
<html>
  <head>
    <title>测试</title>
  </head>
  <body>
    <div class="useful">
      <ul>
        <li class="info">我需要的信息1</li>
        <li class="info">我需要的信息2</li>
        <li class="info">我需要的信息3</li>
      </ul>
     </div>
     <div class="useless">
       <ul>
         <li class="info">垃圾1</li>
         <li class="info">垃圾2</li>
       </ul>
     </div>
  </body>
</html>
'''


# In[7]:


from lxml import html
selector=html.fromstring(source)
info=selector.xpath('//div[@class="useful"]/ul/li/text()')
print(info)


# 豆瓣爬虫

# In[10]:


import requests
import lxml.html
import csv

doubanUrl='https://movie.douban.com/top250?start{}&filter='


# In[11]:


def getSource(url):
    '''
    获取网页源代码
    :param url:
    :return:String
    '''
    head={'use-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.108 Safari/537.36'}
    content=requests.get(url,headers=head)
    content.encoding='utf-8'#强制修改编码，防止windows下出现乱码
    return content.content

def getEveryItem(source):
    '''
    获取每一部电影的相关信息，movie_dict字典用于保存电影的信息
    :param source:
    :return:[movie1_dict,movie2_dict,movie3_dict,...]
    '''
    selector=lxml.html.document_fromstring(source)
    movieItemList=selector.xpath('//div[@class="info"]')#此处使用了先抓大再抓小的技巧
    movieList=[]
    
    for eachMovie in movieList:
        movieList={}
        title-eachMovie.xpath('div[@class="hd"]/a/span[@class="title"]/text()')
        print(title)
        otherTitle=eachMovie.xpath('div[@class="hd"]/a/span[@class="other"]/text()')
        link=eachMovie.xpath('div[@class="hd"]/a/@href')[0]
        directorAndActor=eachMovie.xpath('div[@class="bd"]/p[@class='']/text()')
        star=eachMovie.xpath('div[@class="bd"]/div[@class="star"]/span[@class="rating_num"]/text()')[0]
        quote=eachMovie.xpath('div[@class="bd"]/p[@class="quote"]/span/text()')
        if quote:
            quote=quote[0]
        else:
            quote=''
            
        movieDict['title'] = ''.join(title + otherTitle)
        movieDict['url'] = link
        movieDict['directorAndActor'] = ''.join(directorAndActor).replace('                            ', '').replace('\r', '').replace('\n', '')
        movieDict['star'] = star
        movieDict['quote'] = quote
        print(movieDict)
        movieList.append(movieDict)
    return movieList

def writeData(movieList):
    with open('doubanMovie_example2.csv', 'w', encoding='UTF-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'directorAndActor', 'star', 'quote', 'url'])
        writer.writeheader()
        for each in movieList:
            print(each)
            writer.writerow(each)
            
if __name__ == '__main__':
    movieList = []
    for i in range(20):
        pageLink = doubanUrl.format(i * 1)
        print(pageLink)
        source = getSource(pageLink)
        movieList += getEveryItem(source)
    print(movieList[:10])
    movieList = sorted(movieList, key=lambda k: k['star'], reverse=True) #根据字典中的key=star这一项的value来排序,倒序。
    writeData(movieList)


# 使用Beautiful Soup4 同样用来从HTML和XML中提取数据，某些方面比xpath易懂，但不如xpath简洁，速度也相较慢  
# 主要使用的是find和find_all方法  
# soup = BeautifulSoup(网页源代码, '解析器'),解释器可以填html.parser,也可以填lxml

# In[14]:


source='''
<html>
  <head>
    <title>测试</title>
  </head>
  <body>
    <div class="useful">
      <ul>
        <li class="info">我需要的信息1</li>
        <li class="info">我需要的信息2</li>
        <li class="info">我需要的信息3</li>
      </ul>
     </div>
     <div class="useless">
       <ul>
         <li class="info">垃圾1</li>
         <li class="info">垃圾2</li>
       </ul>
     </div>
  </body>
</html>
'''


# In[15]:


from bs4 import BeautifulSoup
soup=BeautifulSoup(source,'lxml')
useful=soup.find('div',class_="useful")
info=useful.find_all('li')
for i in info:
    print(i.string)
    
#find_all与find的不同在于，find_all返回的是列表，如果没有找到，就会返回空列表；
#find返回的直接是一个BeautifulSoup Tag对象，如果有多个符合条件的BeautifulSoup Tag，则返回第一个，找不到则返回None。
#两者参数完全相同：
    #find_all( name , attrs , recursive , text , **kwargs )
    #name就是HTML的标签名，类似于body, div, ul, li之类。
    #attrs参数的值是一个字典，字典的key是属性名，字典的value是属性值：>>> find_all(attrs={'class': 'useful'})
    #recursive的值为True或者False，当它为False的时候，Beautiful Soup不会搜索子节点。
    #text可以是一个字符串或者是正则表达式。用于搜索标签里面的文本信息：
    #**kwargs表示key=value形式的参数。一般这里的key是属性，value是属性值。
        #这个大多数情况下与标签配合使用，但是有时候如果属性值非常特殊，也可以单独使用：
        #find_all('div', id='test')
        #find_all(class_='iamstrange')
        #需要注意的是，如果某个标签的属性是class的话，在参数里面需要在class后面加一个下划线，写成“class_”避免和Python的关键字class冲突：
        #find_all('div', class_='useful')


# 使用多进程进行检测

# In[16]:


from multiprocessing.dummy import Pool as ThreadPool
import threading
import requests
import time
 
 
def getsource(url):
    global hasViewed #修改全局变量,需要使用global关键字
    print(url)
    html = requests.get(url)
    hasViewed.append(url)
 
 
def threadGetSource(url):
    print(url)
    x = threading.Thread(target=getsource, args=(url,)) #这里的args=(url,) 逗号是必须的,因为加了逗号才是只有一个元素的元组
    x.start()
 
urls = []
hasViewed = []
 
for i in range(1, 21):
    newpage = 'http://tieba.baidu.com/p/3522395718?pn={}'.format(i)
    urls.append(newpage)
 
#==========单线程=====================================
time1 = time.time()
hasViewed = []
for i in urls:
    getsource(i)
print('单线程耗时：{}'.format(time.time() - time1))
 
#============Python multiprocessing.dumpy多线程=======
pool = ThreadPool(20)
time3 = time.time()
hasViewed = []
results = pool.map(getsource, urls)
pool.close()
pool.join()
print('并行耗时：{}'.format(time.time() - time3))
 
#======Python threading 多线程=======================
time5 = time.time()
hasViewed = []
[threadGetSource(url) for url in urls]
while not len(hasViewed) == 20:
    pass
print('thread并行耗时: {}'.format(time.time() - time5))


# Ajax技术介绍  
# AJAX 是 Asynchronous JavaScript And XML 的首字母缩写，意为：异步JavaScript与XML。   
# 使用Ajax技术，可以在不刷新网页的情况下，更新网页数据。  
# 使用Ajax技术的网页，一般会使用HTML编写网页的框架，在打开网页的时候，首先加载的是这个框架。  
# 剩下的部分或者全部内容，将会在框架加载完成以后再通过JavaScript在后台加载。  
# JSON  
# JSON的全称是JavaScript Object Notation，它是一种轻量级的数据交换格式。  
# 我们在网络之间传递数据的时候，绝大多数情况下都是传递的字符串。  
# 当我们需要把Python里面数据发送给网页或者其他的编程语言的时候，可以先将Python的数据转化为JSON格式的字符串，然后将字符串传递给其他的语言，其他语言再将JSON格式的字符串转化为它自己的数据格式。

# In[ ]:


#初始化Selenium
from selenium import webdriver
driver=webdriver.Chrome(r'D:\pydata\chromedriver.exe')
# driver.get("http://v.youku.com/v_show/id_XMTY2NTk5ODAwMA==.html?from=y1.3-idx-beta-1519-23042.223465.3-3")


# 网页动态加载的内容经过加密后，我们是无法看懂密文的，可以用JavaScript解析，用Selenium来模拟浏览器解析JavaScript，再爬取被解析后的代码  

# In[17]:


#等待信息出现
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
WebDriverWait(driver,300).until(EC.presence_of_all_elements_located((By.CLASS_NAME,"con")))
#WebDriverWait会阻塞程序额运行，并每0.5秒检查一次网页源代码。看需要的内容是否出现，如果没有就继续等待。
# 在上面的代码中，设定了超时时间为300秒。在300秒内，如果有某个元素出现，那么就解除阻塞，继续运行后面的代码；
# 如果等待的内容始终不出现，那么就会抛出一个超时的Exception。
EC.presence_of_element_located((By.CLASS_NAME, "con"))
# 这里的EC其实就是 expected_conditions，也就是期望的条件。>>>期望的条件.元素出现
# 而这里的元素就是一个class="con"的元素。
# 这里除了指定class以外，还可以指定很多其他的属性，例如：
By.ID
By.NAME
By.XPATH
# 通过元组的形式传递给presence_of_element_located方法。
# 在网页中寻找我们需要的内容，可以使用类似与Beautiful Soup4 的语法：
element = driver.find_element_by_id("passwd-id") #如果有多个符合条件的，返回第一个
element = driver.find_element_by_name("passwd") #如果有多个符合条件的，返回第一个
element_list = driver.find_elements_by_id("passwd-id") #以列表形式返回所有的符合条件的element
element_list = driver.find_elements_by_name("passwd") #以列表形式返回所有的符合条件的element
# 也可以使用XPath：
element = driver.find_element_by_xpath("//input[@id='passwd-id']") #如果有多个符合条件的，返回第一个
element = driver.find_element_by_xpath("//input[@id='passwd-id']") #以列表形式返回所有的符合条件的element
# 但是有一点需要特别注意：这些名字都是find_element开头的，因此他们返回的都是element对象。这些方法他们的目的是寻找element，而不是提取里面的值。


# 模拟登陆知乎

# In[ ]:


import requests
data={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36'}
source=requests.get('http://zhihu.com',headers=data).text
# print(source)

# 使用selenium来进行模拟登陆。流程如下：
# 初始化chromedriver
# 打开知乎登录页面
# 找到用户名的输入框，输入用户名
# 找到密码输入框，输入用户名
# 按下回车键

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
user_name='******@qq.com'
pass_word='******'
driver=webdriver.Chrome(r'D:\pydata\chromedriver.exe')#填写chromedriver路径
driver.get("https://www.zhihu.com/#signin")

elem=driver.find_element_by_class_name('account')#寻找账号输入框
elem.clear()
elem.send_keys(user_name)#输入账号
password=driver.find_element_by_class_name('password')#寻找密码输入框
password.clear()
password.send_keys(pass_word)#输入密码
elem.send_keys(Keys.RETURN)#模拟键盘回车键
time.sleep(10)#可以直接sleep，也可以等待某个条件出现
print(driver.page_source)
driver.close()#在视频中，这一行会被注释掉，以方便观察结果


# 机锋网爬虫  

# In[ ]:


login_url = 'http://my.gfan.com/login?'
email = '******@163.com'
password = '******'
 
data = {'gotoUrl': 'http://my.gfan.com/',
        'loginName': email,
        'password': password}
 
header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36',
          'Accept': 'text / html, application / xhtml + xml, application / xml;q = 0.9, image / webp, * / *;q = 0.8',
          'Accept-Encoding': 'gzip, deflate, sdch, br',
          'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6'}
 
session = requests.Session()
html = session.get(login_url, headers=header).content.decode()
 
captcha_url = 'http://my.gfan.com/captcha'
 
with open('captcha.png', 'wb') as f:
    f.write(requests.get(captcha_url).content)
 
captcha_solution = input('captcha code is:')
data['authCode'] = captcha_solution
 
result = session.post(login_url, data=data, headers=header).content
print(result.decode())

# 手动输入验证码的一般流程如下：  
# 随机提交数据，获取POST的数据内容和格式  
# 分析网页源代码，获取验证码地址  
# 下载验证码到本地  
# 打开验证码，人眼读取内容  
# 构造POST的数据，填入验证码  
# POST提交

