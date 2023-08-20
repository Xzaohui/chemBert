from transformers import AutoTokenizer, AutoModel
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
data_choose='> <PUBCHEM_CACTVS_TPSA>'
data_choose='D_MS刚性-C3H8'
model_name='TPSA'
model_name='A-SUB'
model_name='COF'
model_name='D-SUB'

criterion = nn.MSELoss()
def test_bert():
    tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/chemBert/model/bertMolE")
    # A-SBU-smile COF-smile D-SBU-smile
    data_frame=data_loader.train_data_bert('/mnt/workspace/chemBert/data/D-SBU-smile.json', tokenizer)
    batch_size=1
    test_dataloader=DataLoader(data_frame[data_choose]['train'],batch_size=batch_size,shuffle=True ,num_workers = 0)
    model=torch.load("/mnt/workspace/chemBert/model/model_{}.pt".format(model_name))
    model.to(device=device)
    model.eval()
    avg_loss=0
    predict=[]
    label=[]
    with torch.no_grad():
        for i,(test_data,attention_mask,test_lab) in enumerate(test_dataloader):
            out=model(test_data.to(device),attention_mask.to(device))
            out=out.to('cpu').squeeze(-1)

            # print(out,test_lab)
            # break
            loss=criterion(out,test_lab)
            if loss.item()>500:
                continue
            # print(loss)
            predict+=list(out.numpy())
            label+=list(test_lab.numpy())
            avg_loss+=loss.item()

    print(r2_score(label,predict))
    print(avg_loss/(i+1))
    print(sum(label)/(i+1))

def test_rf():

    forest = RandomForestRegressor(n_estimators=500,random_state=1,n_jobs=-1)


test_bert()


