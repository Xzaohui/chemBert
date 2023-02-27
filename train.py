from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from model import cheBerta
import torch.nn as nn
import data_loader
import torch
from torch.utils.data import Dataset, DataLoader
import datetime

device='cuda:7'
batch_size=8

train_dataloader=DataLoader(data_loader.data_frame['NR-AR']['train'],batch_size=batch_size,shuffle=True ,num_workers = 0)
dev_dataloader=DataLoader(data_loader.data_frame['NR-AR']['dev'],batch_size=batch_size,shuffle=True ,num_workers = 0)
test_dataloader=DataLoader(data_loader.data_frame['NR-AR']['test'],batch_size=batch_size,shuffle=True ,num_workers = 0)

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model = cheBerta(AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM"))
model.to(device=device)

criterion = nn.CrossEntropyLoss()

# print(criterion(torch.tensor([[0.1,0.9],[0.1,0.9]]),torch.tensor([0,1])))



def train(epoch=1):
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
            if i% 10==0:
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
                    torch.save(model, './model/model.pt')
                    tokenizer.save_pretrained('./model')
                else:
                    print('best:'+str(best))


                





if __name__=='__main__':
    # print(int(1==1))
    train()
