import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font",family="Times New Roman",size=13)
# 数据
data = {
    "MRR": [0.191, 0.209, 0.216, 0.212, 0.261, 0.308],
    "Hits@10": [0.379, 0.399, 0.399, 0.403, 0.447, 0.628],
}

# 模型名称
models = ["DistMult", "TransE", "ConvE", "RotatE", "KG-Predict", "DREG"]

# 条形图位置
x = np.arange(len(models))

# 条形图宽度
width = 0.35

# 创建子图
fig, ax = plt.subplots()

# 绘制MRR条形图（使用阴影）
mrr_bars = ax.bar(x - width/2, data["MRR"], width, label="MRR", color='lightgray', edgecolor='black', linewidth=1.5,hatch='///')

# 绘制Hits@10条形图（使用灰色填充色）
hits_bars = ax.bar(x + width/2, data["Hits@10"], width, label="Hits@10", color='gray', edgecolor='black')

# 设置x轴标签
ax.set_xticks(x)
ax.set_xticklabels(models)

# 添加图例
ax.legend()

# 添加标题和标签
ax.set_title("MRR and Hits@10 for Different Models")
ax.set_xlabel("Models")
ax.set_ylabel("Scores")

# 显示图形
plt.tight_layout()
plt.show()