import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# 假设你已经有一个MRNDR模型的定义
# class MRNDRModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MRNDRModel, self).__init__()
#         self.cos = torch.nn.CosineSimilarity()
#         self.multihead_attn = nn.MultiheadAttention(hidden_dim, 2)
#         self.parameter = torch.nn.Parameter(torch.Tensor([0.5, 0.5]))
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
class MRNDRModel(nn.Module):
    def __init__(self, dict_index, embed_dim, num_heads):
        super(MRNDRModel, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
        self.cos = torch.nn.CosineSimilarity()
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear_2 = torch.nn.Linear(embed_dim, 1)
        self.parameter = torch.nn.Parameter(torch.Tensor([0.5, 0.5]))

    def forward(self, DPTD_name_1, DPTD_name_2):
        DPTD_embed_1 = self.embedding(torch.LongTensor([DPTD_name_1]).unsqueeze(1))
        DPTD_embed_2 = self.embedding(torch.LongTensor([DPTD_name_2]).unsqueeze(1))
        DPTD_embeds_1 = self.linear(DPTD_embed_1)
        DPTD_embeds_2 = self.linear(DPTD_embed_2)
        DPTD_embeds_2 = torch.tanh(DPTD_embeds_2)
        DPTD_embeds_3, _ = self.multihead_attn(DPTD_embeds_1.permute(1, 0, 2), DPTD_embeds_2.permute(1, 0, 2),DPTD_embeds_2.permute(1, 0, 2))
        DPTD_embeds_3 = DPTD_embeds_3.permute(1, 0, 2)
        DPTD_embeds = DPTD_embeds_3 + DPTD_embeds_2
        cos = self.cos(DPTD_embeds_1, DPTD_embeds)
        dist = torch.dist(DPTD_embeds_1, DPTD_embeds)
        oula_loss = self.parameter[0] * cos + self.parameter[1] * dist
        loss = torch.sum(oula_loss)
        return loss, DPTD_embeds_1, DPTD_embeds
# 定义模型超参数
input_dim = 2  # 输入维度，这里假设为2
hidden_dim = 50  # 隐藏层维度，可以根据需要调整
output_dim = 1  # 输出维度，根据任务而定

# 创建MRNDR模型实例
model = MRNDRModel(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

all_data=pd.read_csv("./drug-disease333.csv").values.tolist()
recall_positions = [10, 20, 50, 100, 150]
recall_rates = defaultdict(list)

for _ in range(10):
    # 随机抽取不同数量的样本
    sample_sizes = [100]  # 不同的抽样大小
    for sample_size in sample_sizes:
        num_samples = min(len(all_data), sample_size)
        sampled_data = random.sample(all_data, num_samples)  # all_data是你的全部数据
        drug_names = [sample[1] for sample in sampled_data]
        disease_names = [sample[3] for sample in sampled_data]
        Drug_dict = {}
        unique_drugs = set([sample[1] for sample in sampled_data])
        for index, drug_name in enumerate(unique_drugs):
            Drug_dict[drug_name] = index
        Disease_dict = {}
        unique_disease = set([sample[3] for sample in sampled_data])
        for index, disease_name in enumerate(unique_disease):
            Disease_dict[disease_name] = index
        drug_indices = [Drug_dict[name] for name in drug_names]
        disease_indices = [Disease_dict[name] for name in disease_names]
        drug_indices_tensor = torch.tensor(drug_indices, dtype=torch.long)
        disease_indices_tensor = torch.tensor(disease_indices, dtype=torch.long)
        sampled_data_tensor = torch.stack((drug_indices_tensor, disease_indices_tensor), dim=1)

        targets = torch.tensor([sample[-1] for sample in sampled_data], dtype=torch.float32)
        for epoch in range(10):  # 假设进行10个epoch的训练
            optimizer.zero_grad()
            outputs = model(sampled_data_tensor.float())
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()

        # 对抽样的数据进行预测并排序
        sampled_data_tensor[:,-1]= model(sampled_data_tensor[:,:].float()).squeeze()
        sampled_data_tensors = sampled_data_tensor.detach().numpy()
        sampled_data_tensors = sampled_data_tensors[sampled_data_tensors.argsort()[::-1]]
        for recall_position in recall_positions:
                # 获取前recall_position个样本
                top_samples = sampled_data_tensors[:recall_position]
                successful_indices = []
                for index, sample in enumerate(top_samples):
                    sample=[i[0] for i in sample]
                    if (sample[0], sample[1]) in sampled_data_tensor.numpy():
                        successful_indices.append(index + 1)

                # 统计验证成功的位置分布
                for position in recall_positions:
                    recall_count = sum(1 for idx in successful_indices if idx <= position)
                    recall_rate = recall_count / len(successful_indices) if successful_indices else 0
                    recall_rates[position].append(recall_rate)
                # 输出结果
# 输出结果
for position, rates in recall_rates.items():
    avg_recall_rate = sum(rates) / len(rates)
    print(f"Average Recall Rate at Position {position}: {avg_recall_rate:.4f}")
