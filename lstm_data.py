from torch.utils.data import Dataset, DataLoader
import torch

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import pandas as pd
from torch.utils.data import Dataset, DataLoader

tokenizer = ByteLevelBPETokenizer(
    "./tokenizer/vocab.json",
    "./tokenizer/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

class dataset(Dataset):
    def __init__(self,total_data,total_lab):
        self.total_data=total_data
        self.total_lab=total_lab
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_lab[idx]

tox=pd.read_csv('tox21.csv',header=0)
# tox=tox.applymap(lambda x: x if str(x) != 'nan' else 2.0)
tox=tox.dropna()

smiles=[list(ids.ids)+[0]*(512-len(ids.ids)) for ids in tokenizer.encode_batch(tox['smiles'])]


data_frame={}
for name in list(tox.columns)[:-2]:
    data_frame[name]={'train':dataset(torch.tensor(smiles).long()[:int(len(tox)*0.8)],torch.tensor(list(tox[name].values)).long()[:int(len(tox)*0.8)]),'dev':dataset(torch.tensor(smiles).long()[int(len(tox)*0.8):int(len(tox)*0.9)],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.8):int(len(tox)*0.9)]),'test':dataset(torch.tensor(smiles).long()[int(len(tox)*0.9):],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.9):])}