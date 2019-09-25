# -*- coding:utf-8 -*-


'''

房天下深圳新房信息

'''

import requests

from bs4 import BeautifulSoup

import numpy as np

import re

URL = 'https://sz.newhouse.fang.com/house/s/'

HTML = requests.get(URL)

SOUP = BeautifulSoup(HTML.content, 'html.parser', from_encoding='gb18030')

last_page = SOUP.select('.last')

page_number = int(last_page[0]['href'].split('/')[3].split('9')[1])#根据尾页划分页码

url_demo = 'https://sz.newhouse.fang.com/house/s/'#i+1,name.text.strip(),

#房价价格

house_price_list=[]

for i in range(1,(page_number+1)):

    url = url_demo.format(i)

    html = requests.get(url)

    soup = BeautifulSoup(html.content,'html.parser',from_encoding='gb18030')

    names = soup.select('.nlcd_name a')#class定位组合查找

    adresses = soup.select('.address a')#查找地址

    all_type = soup.findAll(name="span", attrs={"class": re.compile(r"forSale|inSale|outSale|zusale|zushou")})#出售

    all_money = soup.findAll(name="div", attrs={"class": re.compile(r"nhouse_price|kanesf")})#价格

    for i,name in enumerate(names):

        print(i+1,' name：'+name.text.strip(),'  address：'+''.join(re.split(r'\s+',

               adresses[i].text.replace('\n','').replace('',''))),

              all_type[i].text,' house_price: '+all_money[i].text.replace('\n',''))

        house_price_list.append(re.findall('\d+',all_money[i].text.replace('\n','')))

house_price_list=[int(i[0]) for i in house_price_list if i]

print('*'*80)

print('* '+' 房价均价:'+str(np.mean(house_price_list))+' '*60+'*')

print('* '+' 房价最高价:'+str(np.max(house_price_list))+' '*60+'*')

print('* '+' 房价最低价:'+str(np.min(house_price_list))+' '*61+'*')

print('*'*80)