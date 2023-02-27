import torch
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self,total_data,total_mask,total_lab):
        self.total_data=total_data
        self.total_mask=total_mask
        self.total_lab=total_lab
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_mask[idx],self.total_lab[idx]
    

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")


tox=pd.read_csv('tox21.csv',header=0)
# tox=tox.applymap(lambda x: x if str(x) != 'nan' else 2.0)
tox=tox.dropna()


smiles=tokenizer(list(tox['smiles'].values),truncation=True,padding="max_length",max_length=512,return_tensors="pt")


data_frame={}
for name in list(tox.columns)[:-2]:
    data_frame[name]={'train':dataset(smiles['input_ids'].long()[:int(len(tox)*0.8)],smiles['attention_mask'].long()[:int(len(tox)*0.8)],torch.tensor(list(tox[name].values)).long()[:int(len(tox)*0.8)]),'dev':dataset(smiles['input_ids'].long()[int(len(tox)*0.8):int(len(tox)*0.9)],smiles['attention_mask'].long()[int(len(tox)*0.8):int(len(tox)*0.9)],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.8):int(len(tox)*0.9)]),'test':dataset(smiles['input_ids'].long()[int(len(tox)*0.9):],smiles['attention_mask'].long()[int(len(tox)*0.9):],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.9):])}




# from transformers import AutoTokenizer, AutoModel

# model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM",num_labels=2)


# print(data_frame['NR-AR']['train'].total_data[0])

# print(model(data_frame['NR-AR']['train'].total_data[3].unsqueeze(0),attention_mask=data_frame['NR-AR']['train'].total_mask[3].unsqueeze(0)).last_hidden_state.shape)






