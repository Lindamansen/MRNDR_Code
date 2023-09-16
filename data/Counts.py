#encoding=utf-8
import pandas as pd
import numpy as np

dataset=pd.read_csv("data_tui_1.csv")

data_drop=dataset.iloc[:,1].drop_duplicates()
print(len(data_drop))
disease_name=[]
data_all=[]

for data in data_drop:
    if "推荐" in data:
        disease_name.append(data.split("疾病")[1])
        data_all.append(data)
dataframe=pd.DataFrame(data_all)
print(dataframe)
dataframe.to_csv(r"C:\Users\Lindamansen\Desktop\disease\data_tui.csv")

# data_ing=pd.value_counts(disease_name)
#
# print(data_ing)
#
# for i in range(len(data_ing)):
#     print(data_ing[i],data_ing.index[i])


