#coding=utf-8
from bs4 import BeautifulSoup
import re
import urllib.request,urllib.error
import urllib.parse
import pandas as pd
import requests


def redata(baseurl):
    datalist=[]
    findvalue = re.compile(r'<span class="value">(.*)</span>')
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    res=urllib.request.Request(url=baseurl,headers=headers)
    response=urllib.request.urlopen(res)
    # print(response.read().decode("utf-8"))
    soup=BeautifulSoup(response, 'lxml')
    item=soup.find_all('div', class_="results-amount")
    # score=soup.find_all('div',class_="docsum-citation full-citation")
    score=soup.find_all('div',class_="docsum-citation full-citation")
    score=str(score)
    item = str(item)
    # print(item)
    if "<span" in item:
        data= re.findall(findvalue,item)[0]
        print(data)
    else:
        data=0
        print(data)
    return data
if __name__ == '__main__':
    sum_data_all=[]
    dataset = pd.read_csv("抽取药物进行调研.csv")
    drug_name =dataset["Drug"]
    disease_name=dataset["Disease"]
    for index,(Drug,Disease) in enumerate(zip(drug_name,disease_name)):
        data=redata("https://pubmed.ncbi.nlm.nih.gov/?term={}+AND+{}&sort=".format(Drug.replace(" ","+"),Disease.replace(" ","+")))
        sum_data_all.append(data)
