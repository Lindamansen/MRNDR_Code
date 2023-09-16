#encoding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc("font",family="Times New Roman",size=24)
# 读取数据
loss_data = pd.read_csv("Loss.csv", encoding="utf-8")

# 提取数据
Epoch = loss_data["Epoch"]
Cos = loss_data["Cos"]
Dist = loss_data["Dist"]
Cos_Dist = loss_data["Cos+Dist"]

# 设置样式
plt.style.use('seaborn-white')  # 使用Seaborn样式
plt.figure(figsize=(10, 7))  # 设置图表尺寸
# plt.plot(Epoch, Cos, label='left part formula', marker='o')
plt.plot(Epoch, Dist, label='Right part formula', marker='s')
plt.plot(Epoch, Cos_Dist, label='EPS algorithm', marker='^')

plt.title('Loss Comparison Over Epochs', fontsize=24)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
# 添加图例
plt.legend(loc="center right")

# 显示图表
# plt.tight_layout()  # 避免标签重叠
plt.show()





