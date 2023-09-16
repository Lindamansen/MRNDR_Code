#encoding=utf-8
import torch
import numpy as np
from dataset import dataset_port,dataset_port_2
from model import matrix_calculate,result_matrix
import pandas as pd
import random
from MRR import Mrr,His
if __name__ == '__main__':
    embed_dim=20
    # data_path=r"./data/drug-disease.csv"
    data_path="../data/DTINet.csv"
    # data_path="../data/Drug-target-disease.csv"
    # data_path = "../data/KG.csv"
    Drug_name,Target_name,Disease_name=dataset_port_2(data_path)
    Drug_result=[]
    Drug_dict= {}
    Target_result = []
    Target_dict = {}
    for i in range(len(Drug_name)):
        Drug_dict[Drug_name[i]]=Drug_name.index[i]
    Drug_index_all=[]
    for i in range(len(Drug_name)):
        Drug_index_all.append(Drug_dict[Drug_name[i]])
    for i in range(len(Target_name)):
        Target_dict[Target_name[i]]=Target_name.index[i]
    Target_index_all=[]
    for i in range(len(Target_name)):
        Target_index_all.append(Target_dict[Target_name[i]])

    result_drug=[]
    Disease_embed_drug=[]

    embedding_model_drug= matrix_calculate(Drug_name)
    optimizer = torch.optim.Adam([embedding_model_drug.parameter],lr=1e-20)
    Mrr_score=[]
    Hits1_score = []
    Hits3_score = []
    Hits10_score = []
    for a in range(2):
        for index,(Drug,Disease,Drug_index) in enumerate(zip(Drug_name,Disease_name,Drug_index_all[:20])):
            optimizer.zero_grad()
            loss,Drug_embed,Disease_embed=embedding_model_drug(Drug_index,Drug_index)
            loss.backward()
            optimizer.step()
            Wr_matrix = torch.rand(embed_dim, 1)
            rand_matrix = torch.mm(Wr_matrix,Drug_embed)
            result=result_matrix(rand_matrix,Wr_matrix,Drug_embed)
            cos=torch.nn.CosineSimilarity()
            cos_result=cos(result,Disease_embed)
            # print(Drug_name[index],"已知治疗",Disease_name[index],cos_result.item())
            result_drug.append(result)
            Disease_embed_drug.append(Disease_embed)
        result_target=[]
        Disease_embed_target=[]
        embedding_model_target= matrix_calculate(Target_name)
        optimizer = torch.optim.Adam([embedding_model_target.parameter], lr=1e-20)
        for index, (Target, Disease, Target_index) in enumerate(zip(Target_name, Disease_name, Target_index_all[:20])):
            optimizer.zero_grad()
            loss, Target_embed, Disease_embed = embedding_model_target(Target_index, Target_index)
            loss.backward()
            optimizer.step()
            Wr_matrix = torch.rand(embed_dim, 1)
            rand_matrix = torch.mm(Wr_matrix, Target_embed)
            result = result_matrix(rand_matrix, Wr_matrix, Target_embed)
            cos = torch.nn.CosineSimilarity()
            cos_result = cos(result, Disease_embed)
            # print(Drug_name[index],"已知治疗",Disease_name[index],cos_result.item())
            result_target.append(result)
            Disease_embed_target.append(Disease_embed)
        #推荐系统
        score_drug = []
        location_drug=[]
        for i,(results_drug) in enumerate(result_drug):
            cos = torch.nn.CosineSimilarity()
            for j,(disease_embed) in enumerate(Disease_embed_drug):
                cos_result = cos(results_drug, disease_embed)
                if 0.6<cos_result.item()<=1:
                    if i==j:
                        # print(Drug_name[i], "已知治疗", Disease_name[j], cos_result.item())
                        score_drug.append(cos_result.item())
                        location_drug.append(j)
                    else:
                        # print(Drug_name[i], "药物推荐治疗疾病", Disease_name[j], cos_result.item())
                        score_drug.append(cos_result.item())
                        location_drug.append(j)
        score_target = []
        location_target=[]
        for i,(results_target) in enumerate(result_target):
            cos = torch.nn.CosineSimilarity()
            for j,(disease_embed) in enumerate(Disease_embed_target):
                cos_result = cos(results_target, disease_embed)
                if 0.6<cos_result.item()<=1:
                    if i==j:
                        # print(Target_name[i], "已知关系", Disease_name[j], cos_result.item())
                        score_target.append(cos_result.item())
                        location_target.append(j)
                    else:
                        # print(Target_name[i], "靶点推荐疾病", Disease_name[j], cos_result.item())
                        score_target.append(cos_result.item())
                        location_target.append(j)

        drug_disease = zip(location_drug, score_drug)
        drug_disease_array = sorted(drug_disease, key=lambda x: x[1], reverse=True)
        target_disease = zip(location_target, score_target)
        target_disease_array= sorted(target_disease, key=lambda x: x[1], reverse=True)
        distance_score=[]
        for index,(drug,target) in enumerate(zip(drug_disease_array,target_disease_array)):
            if Disease_name[drug[0]]==Disease_name[target[0]]:
                # print(drug[0]+1,target[0]+1)
                distance_abs=abs((drug[0]+1)-(target[0]+1))
                if distance_abs==0:
                    distance_score.append(1)
                else:
                    distance_score.append(distance_abs)
                # print(drug[0]+1,target[0]+1)
        # for index, (drug, target) in enumerate(zip(drug_disease_array, target_disease_array)):
        #     if Disease_name[drug[0]] == Disease_name[target[0]]:
        #         distance_abs = ((drug[0] + 1) + (target[0] + 1))/2
        #         print(distance_abs)
        #         distance_score.append(distance_abs)
        print(Mrr(len(distance_score), distance_score))
        Mrr_score.append(Mrr(len(distance_score), distance_score))
        # print(Mrr(len(distance_score),distance_score))
        print(His(len(distance_score), 1, distance_score))
        Hits1_score.append(His(len(distance_score), 1, distance_score))
        print(His(len(distance_score), 3, distance_score))
        Hits3_score.append(His(len(distance_score), 3, distance_score))
        print(His(len(distance_score), 10, distance_score))
        Hits10_score.append(His(len(distance_score), 10, distance_score))
    print("MRR:",max(Mrr_score))
    print("Hits_1:",max(Hits1_score))
    print("Hits_3:",max(Hits3_score))
    print("Hits_10:", max(Hits10_score))
    # print("MRR:", sum(Mrr_score)/len(Mrr_score))
    # print("Hits_1:", sum(Hits1_score)/len(Mrr_score))
    # print("Hits_3:", sum(Hits3_score)/len(Mrr_score))
    # print("Hits_10:", sum(Hits10_score)/len(Mrr_score))
