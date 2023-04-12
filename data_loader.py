import torch
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import random


class dataset(Dataset):
    def __init__(self,total_data,total_mask,total_lab):
        self.total_data=total_data
        self.total_mask=total_mask
        self.total_lab=total_lab
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_mask[idx],self.total_lab[idx]

data_frame={}



def train_data_manage():
    tox=pd.read_csv('tox21.csv',header=0)
    # tox=tox.applymap(lambda x: x if str(x) != 'nan' else 2.0)
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    tox=tox.dropna()
    smiles=tokenizer(list(tox['smiles'].values),truncation=True,padding="max_length",max_length=512,return_tensors="pt")
    for name in list(tox.columns)[:-2]:
        data_frame[name]={'train':dataset(smiles['input_ids'].long()[:int(len(tox)*0.8)],smiles['attention_mask'].long()[:int(len(tox)*0.8)],torch.tensor(list(tox[name].values)).long()[:int(len(tox)*0.8)]),'dev':dataset(smiles['input_ids'].long()[int(len(tox)*0.8):int(len(tox)*0.9)],smiles['attention_mask'].long()[int(len(tox)*0.8):int(len(tox)*0.9)],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.8):int(len(tox)*0.9)]),'test':dataset(smiles['input_ids'].long()[int(len(tox)*0.9):],smiles['attention_mask'].long()[int(len(tox)*0.9):],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.9):])}

def random_mask(ids):
    for i in range(len(ids)):
        if ids[i] in [0,1,2,4]:
            continue
        if torch.rand(1)>0.85:
            if torch.rand(1)>0.8:
                ids[i]=4
            else:
                ids[i]=random.randint(5,9999)
    return ids

pretarin_data={}
def pretrain_data_manage():
    f=open('smiles.txt','r',encoding='utf-8')
    smiles=f.readlines()
    smiles=[s.replace('\n','') for s in smiles]
    tokenizer = ByteLevelBPETokenizer(
        "./tokenizer/vocab.json",
        "./tokenizer/merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>", pad_to_multiple_of=512)

    smiles=tokenizer.encode_batch(smiles)

    print(random_mask(smiles[0].ids))
    






# from transformers import AutoTokenizer, AutoModel

# model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM",num_labels=2)


# print(data_frame['NR-AR']['train'].total_data[0])

# print(model(data_frame['NR-AR']['train'].total_data[3].unsqueeze(0),attention_mask=data_frame['NR-AR']['train'].total_mask[3].unsqueeze(0)).last_hidden_state.shape)

if __name__=='__main__':
    pretrain_data_manage()






