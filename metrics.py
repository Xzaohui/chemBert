from transformers import RobertaTokenizer, AutoModel
from model import chemBert_c,chemBert_r
import torch.nn as nn
import data_loader
import torch
from torch.utils.data import DataLoader
import datetime
from model import lstm
import lstm_data
import data_loader
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor #导入随机森林模型

device='cuda'






def test_bert():
    data_frame=data_loader.train_data_bert()
    batch_size=8
    test_dataloader=DataLoader(data_frame[data_loader.data_choose]['test'],batch_size=batch_size,shuffle=True ,num_workers = 0)
    model=torch.load(r"model\model_bert\model.pt")
    model.to(device=device)
    model.eval()
    avg_loss=0
    predict=[]
    label=[]
    with torch.no_grad():
        for i,(test_data,attention_mask,test_lab) in enumerate(test_dataloader):
            out=model(test_data.to(device),attention_mask.to(device))
            out=out.to('cpu').squeeze(-1)
            predict+=list(out.numpy())
            label+=list(test_lab.numpy())
            criterion = nn.MSELoss()
            loss=criterion(out,test_lab)
            avg_loss+=loss.item()
    print(r2_score(label,predict))
    print(avg_loss/(i+1))

def test_rf():

    forest = RandomForestRegressor(n_estimators=500,random_state=1,n_jobs=-1)


test_bert()


