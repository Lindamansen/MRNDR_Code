#encoding=utf-8
import numpy as np
import torch

def calculate_mrr(ranks):
    # 计算Mean Reciprocal Rank (MRR)
    mrr = torch.reciprocal(ranks.float()).mean().item()
    return mrr

def calculate_hits(ranks, k):
    # 计算Hits@k
    hits = (ranks <= k).float().mean().item()
    return hits
