#encoding=utf-8
import pandas as pd

def dataset_port(data_path):
    data=pd.read_csv(data_path)
    Drug_name=data["Drug_name"]
    Disease_name=data["Disease_name"]
    return Drug_name,Disease_name

def dataset_port_2(data_path):
    data=pd.read_csv(data_path)
    Drug_name=data["Drug_name"]
    Target_name=data["Target_name"]
    Disease_name=data["Disease_name"]
    return Drug_name,Target_name,Disease_name
