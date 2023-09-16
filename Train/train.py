#encoding=utf-8
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score
import random
data = pd.read_csv("DTINet.csv")

# Create dictionaries for Drug_id and Disease_id
drug_dict = {drug_id: idx for idx, drug_id in enumerate(data["Drug_name"].unique())}
disease_dict = {disease_id: idx for idx, disease_id in enumerate(data["Disease_name"].unique())}

# Convert Drug_id and Disease_id to indices
data["Drug_idx"] = data["Drug_name"].map(drug_dict)
data["Disease_idx"] = data["Disease_name"].map(disease_dict)

# Define a custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        DPTD_name_1 = self.data.iloc[index]["Drug_idx"]
        DPTD_name_2 = self.data.iloc[index]["Disease_idx"]
        return DPTD_name_1, DPTD_name_2

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create DataLoader for training and testing sets
batch_size = 64
train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the model
class matrix_calculate(torch.nn.Module):
    def __init__(self, dict_index, embed_dim):
        super(matrix_calculate, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
        self.cos = torch.nn.CosineSimilarity()
        self.linear = torch.nn.Linear(embed_dim, 10)
        self.linear_2 = torch.nn.Linear(10, 1)
        self.parameter = torch.nn.Parameter(torch.Tensor([0.5, 0.5]))

    def forward(self, DPTD_name_1, DPTD_name_2):
        DPTD_embed_1 = self.embedding(DPTD_name_1)
        DPTD_embed_2 = self.embedding(DPTD_name_2)
        DPTD_embeds_1 = self.linear(DPTD_embed_1)
        DPTD_embeds_2 = self.linear(DPTD_embed_2)
        DPTD_embeds_2 = torch.tanh(DPTD_embeds_2)
        DPTD_embeds_3 = self.linear_2(DPTD_embeds_2)
        DPTD_embeds = DPTD_embeds_3 + DPTD_embeds_2
        cos = self.cos(DPTD_embeds_1, DPTD_embeds)
        dist = torch.dist(DPTD_embeds_1, DPTD_embeds)
        Similarities = self.parameter[0] * cos + self.parameter[1] * dist
        return Similarities, DPTD_embeds_1, DPTD_embeds

def train_model(model, optimizer, data_loader, num_epochs):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for DPTD_name_1, DPTD_name_2 in data_loader:
            optimizer.zero_grad()
            sim,_,_ = model(DPTD_name_1, DPTD_name_2)
            target = torch.ones_like(sim)
            loss = criterion(sim, target)
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += 1e-5 * l2_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.3f}")

# def evaluate(model, data_loader, top_k=[1, 5, 10]):
#     model.eval()
#     with torch.no_grad():
#         ranks = []
#         for DPTD_name_1, DPTD_name_2 in data_loader:
#             similarity, _,_ = model(DPTD_name_1, DPTD_name_2)
#             _, sorted_indices = torch.sort(similarity, descending=True)
#             rank = (sorted_indices == 0).nonzero().item() + 1
#             ranks.append(rank)
#
#         mrr = np.mean(1.0 / np.array(ranks))
#         print(f"MRR: {mrr:.4f}")
#         for k in top_k:
#             hits_k = np.mean(np.array(ranks) <= k)
#             print(f"Hits@{k}: {hits_k:.4f}")
def evaluate_until_threshold(model, data_loader, top_k=[1, 5, 10], mrr_threshold=0.35, hits_1_threshold=0.35, hits_5_threshold=0.5, hits_10_threshold=0.7):
    model.eval()
    with torch.no_grad():
        ranks = []
        for DPTD_name_1, DPTD_name_2 in data_loader:
            similarity_pos, _, _ = model(DPTD_name_1, DPTD_name_2)

            _, sorted_indices_pos = torch.sort(similarity_pos, descending=True)
            rank_pos = (sorted_indices_pos == 0).nonzero().item() + 1
            ranks.append(rank_pos)

            for _ in range(5):
                DPTD_name_1_neg = DPTD_name_1
                DPTD_name_2_neg = torch.randint_like(DPTD_name_2,high=len(data_loader.dataset.data["Disease_idx"].unique()))
                similarity_neg, _, _ = model(DPTD_name_1_neg, DPTD_name_2_neg)
                _, sorted_indices_neg = torch.sort(similarity_neg, descending=True)
                rank_neg = (sorted_indices_neg == 0).nonzero().item() + 1
                ranks.append(rank_neg)

        mrr = np.mean(1.0 / np.array(ranks))
        print(f"MRR: {mrr:.4f}")
        hits_k = [np.mean(np.array(ranks) <= k) for k in top_k]
        for k, hits in zip(top_k, hits_k):
            print(f"Hits@{k}: {hits:.4f}")
def evaluate(model, data_loader, top_k=[1, 5, 10]):
    model.eval()
    with torch.no_grad():
        all_similarities = []
        for DPTD_name_1, DPTD_name_2 in data_loader:
            similarities, _, _ = model(DPTD_name_1, DPTD_name_2)
            all_similarities.append(similarities.cpu().numpy())

        # all_similarities = np.concatenate(all_similarities)
        # Calculate Mean Average Precision (MAP)
        avg_precision = average_precision_score(np.ones_like(all_similarities), all_similarities)

        # Calculate Normalized Discounted Cumulative Gain (NDCG) at different cutoffs
        sorted_indices = np.argsort(all_similarities)[::-1]
        ndcg_values = []
        for k in top_k:
            ideal_dcg = np.sum(1.0 / np.log2(np.arange(2, k + 2)))
            dcg = np.sum(1.0 / np.log2(np.arange(2, sorted_indices.shape[0] + 2)) * (np.ones_like(sorted_indices) / np.log2(sorted_indices + 2)))
            ndcg = dcg / ideal_dcg
            ndcg_values.append(ndcg)

        print(f"Mean Average Precision (MAP): {avg_precision:.4f}")
        for k, ndcg in zip(top_k, ndcg_values):
            print(f"NDCG@{k}: {ndcg:.4f}")
# Create the model and optimizer
embed_dim = 128  # Replace with your desired embedding dimension
model = matrix_calculate(disease_dict, embed_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
train_model(model, optimizer, train_loader, num_epochs)

# Evaluate the model
# evaluate(model, test_loader)
evaluate_until_threshold(model, test_loader)