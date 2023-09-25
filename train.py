from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from model import chemBert_c,chemBert_r
import torch.nn as nn
import data_loader
import torch
from torch.utils.data import DataLoader
import datetime
from model import lstm
# import lstm_data
import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.externals import joblib

device='cpu'
batch_size=1
tokenizer = AutoTokenizer.from_pretrained("./models/chembert")
data_choose='> <PUBCHEM_CACTVS_TPSA>'
data_choose='formation_energy'
model_name='formation_energy'

# data_frame=data_loader.train_data_tox()
# train_dataloader=DataLoader(data_frame['NR-AR']['train'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# dev_dataloader=DataLoader(data_frame['NR-AR']['dev'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# test_dataloader=DataLoader(data_frame['NR-AR']['test'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# model = chemBert_c(AutoModel.from_pretrained(""))

# A-SBU-smile COF-smile D-SBU-smile
data_frame=data_loader.train_data_bert('./data/crystal.json', tokenizer)
train_dataloader=DataLoader(data_frame[data_choose]['train'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# dev_dataloader=DataLoader(data_frame[data_choose]['dev'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# m=AutoModel.from_pretrained("/mnt/workspace/chemBert/model/ChemBERTa-77M-MLM")
m=torch.load('./models/chembert/chembert.pt')
model = chemBert_r(m)

# train_dataloader=DataLoader(lstm_data.data_frame['NR-AR']['train'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# dev_dataloader=DataLoader(lstm_data.data_frame['NR-AR']['dev'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# test_dataloader=DataLoader(lstm_data.data_frame['NR-AR']['test'],batch_size=batch_size,shuffle=True ,num_workers = 0)
# model=lstm()

model.to(device=device)

# criterion = nn.CrossEntropyLoss()

criterion = nn.MSELoss()


def train_bert(epoch=1):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06*total_steps,num_training_steps=total_steps)

    best=999
    for _ in range(epoch):
        for i,(train_data,attention_mask,train_lab) in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()
            out=model(train_data.to(device),attention_mask.to(device))
            loss=criterion(out.squeeze(-1),train_lab.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i%10==0:
                print(loss)
                
                # dev_num=len(data_frame['> <PUBCHEM_EXACT_MASS>']['dev'])
                # error_list=[]
                # model.eval()
                # for dev_data,attention_mask,dev_lab in dev_dataloader:
                #     out=model(dev_data.to(device),attention_mask.to(device))
                #     out=out.to('cpu').unsqueeze(-1)
                #     error_list.append(criterion(out,dev_lab).item())
                # print("=================================================")
                # print(datetime.datetime.now)
                # print('epoch:'+str(e)+'step:'+str(i))
                # if sum(error_list)/dev_num < best:
                #     print('now_best:'+str(best)+'->'+str(sum(error_list)/dev_num))
                #     best=sum(error_list)/dev_num
                #     torch.save(model, './model/model_bert/model.pt')
                #     tokenizer.save_pretrained('./model/model_bert')
                # else:
                #     print('best:'+str(best))
    torch.save(model, './models/model_{}.pt'.format(model_name))

def train_bert_toxic(epoch=1):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06*total_steps,num_training_steps=total_steps)

    best=0
    for e in range(epoch):
        for i,(train_data,attention_mask,train_lab) in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()
            out=model(train_data.to(device),attention_mask.to(device))
            loss=criterion(out,train_lab.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i%10==0:
                print(loss)
            if i%100==0:
                dev_num=len(data_loader.data_frame['NR-AR']['dev'])
                correct_num=0.0
                model.eval()
                for dev_data,attention_mask,dev_lab in dev_dataloader:
                    out=model(dev_data.to(device),attention_mask.to(device))
                    predict_lab=torch.max(out,1).indices.to('cpu').tolist()
                    dev_lab=dev_lab.tolist()
                    for b in range(len(predict_lab)):
                        correct_num+=int(predict_lab[b]==dev_lab[b])
                print("=================================================")
                print(datetime.datetime.now)
                print('epoch:'+str(epoch)+'step:'+str(i))
                if correct_num/dev_num > best:
                    print('now_best:'+str(best)+'->'+str(correct_num/dev_num))
                    best=correct_num/dev_num
                    torch.save(m, './models/model.pt')
                    tokenizer.save_pretrained('./model/model_bert')
                else:
                    print('best:'+str(best))


                
def train_lstm(epoch=1):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06*total_steps,num_training_steps=total_steps)

    best=0
    best=0
    for e in range(epoch):
        for i,(train_data,train_lab) in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()
            out=model(train_data.to(device))
            loss=criterion(out,train_lab.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i% 10==0:
                print(loss)
            if i%100==0:
                dev_num=len(data_loader.data_frame['NR-AR']['dev'])
                correct_num=0.0
                model.eval()
                for dev_data,dev_lab in dev_dataloader:
                    out=model(dev_data.to(device))
                    predict_lab=torch.max(out,1).indices.to('cpu').tolist()
                    dev_lab=dev_lab.tolist()
                    for b in range(len(predict_lab)):
                        correct_num+=int(predict_lab[b]==dev_lab[b])
                print("=================================================")
                print(datetime.datetime.now)
                print('epoch:'+str(epoch)+'step:'+str(i))
                if correct_num/dev_num > best:
                    print('now_best:'+str(best)+'->'+str(correct_num/dev_num))
                    best=correct_num/dev_num
                    torch.save(model.state_dict(), './model/model_lstm/model.pkl')
                else:
                    print('best:'+str(best))

def train_rf():
    train_data=data_loader.train_data_rf()
    forest = RandomForestRegressor(n_estimators=500,random_state=1,n_jobs=-1)
    forest.fit(train_data[data_loader.data_choose][0],train_data[data_loader.data_choose][1])
    joblib.dump(forest, './model/model_rf/model.pkl')
    print(forest.score(train_data['train'][data_loader.data_choose][0],train_data['train'][data_loader.data_choose][1]))
    print(forest.score(train_data['test'][data_loader.data_choose][0],train_data['test'][data_loader.data_choose][1]))



if __name__=='__main__':
    # train_toxic()
    # train_lstm()
    train_bert(20)
    # train_rf()
