#encoding=utf-8
import torch
import numpy as np
from dataset import dataset_port
from model import matrix_calculate,result_matrix
import pandas as pd
import random

if __name__ == '__main__':
    embed_dim=20
    data_path=r"./data/drug-disease.csv"
    # data_path="./data/drug_disease_DTINet.csv"
    Drug_name,Disease_name=dataset_port(data_path)
    Drug_result=[]
    Drug_dict= {}
    Disease_result = []
    Disease_dict = {}
    result_dict=[]
    for i in range(len(Drug_name)):
        Drug_dict[Drug_name[i]]=Drug_name.index[i]

    Drug_index_all=[]
    for i in range(len(Drug_name)):
        Drug_index_all.append(Drug_dict[Drug_name[i]])
    result_all=[]
    Disease_embed_all=[]

    embedding_model = matrix_calculate(Drug_name)
    optimizer = torch.optim.Adam([embedding_model.parameter],lr=1e-20)
    # random_number = [random.randint(0, 108344) for i in range(10)]
    for index,(Drug,Disease,Drug_index) in enumerate(zip(Drug_name,Disease_name,Drug_index_all)):
        optimizer.zero_grad()
        loss,Drug_embed,Disease_embed=embedding_model(Drug_index,Drug_index)
        loss.backward()
        optimizer.step()
        Wr_matrix = torch.rand(embed_dim, 1)
        rand_matrix = torch.mm(Wr_matrix,Drug_embed)
        result=result_matrix(rand_matrix,Wr_matrix,Drug_embed)
        cos=torch.nn.CosineSimilarity()
        cos_result=cos(result,Disease_embed)
        # print(Drug_name[index],"已知治疗",Disease_name[index],cos_result.item())
        result_all.append(result)
        Disease_embed_all.append(Disease_embed)
    #推荐系统
    score = []
    name = []
    for i,(result_drug) in enumerate(result_all):
        cos = torch.nn.CosineSimilarity()
        for j,(disease_embed) in enumerate(Disease_embed_all):
            cos_result = cos(result_drug, disease_embed)
            if 0.85<cos_result.item()<=1:
                if i==j:
                    print(Drug_name[i], "已知治疗", Disease_name[j], cos_result.item())
                    name.append(Drug_name[i]+"药物已知治疗疾病"+Disease_name[j])
                    score.append(cos_result.item())
                else:
                    print(Drug_name[i], "药物推荐治疗疾病", Disease_name[j], cos_result.item())
                    name.append(Drug_name[i]+"药物推荐治疗疾病"+Disease_name[j])
                    score.append(cos_result.item())
    drug_disease= zip(name, score)
    drug_disease_array = sorted(drug_disease, key=lambda x: x[1], reverse=True)
    print(drug_disease_array)
    # pd.DataFrame(drug_disease_array).to_csv("./dataset/data_pyqt5/name_text.csv", mode="w")