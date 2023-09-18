import torch
from transformers import RobertaTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import random

data_title=['formation_energy','D_MS刚性-C3H8', '> <PUBCHEM_XLOGP3_AA>','> <PUBCHEM_EXACT_MASS>','> <PUBCHEM_MOLECULAR_WEIGHT>','> <PUBCHEM_CACTVS_TPSA>','> <PUBCHEM_MONOISOTOPIC_WEIGHT>','> <PUBCHEM_TOTAL_CHARGE>','> <PUBCHEM_HEAVY_ATOM_COUNT>']
# XLOGP3疏水性
# TPSA分子极性表面积
# CHARGE电荷
class dataset(Dataset):
    def __init__(self,total_data,total_mask,total_lab):
        self.total_data=total_data
        self.total_mask=total_mask
        self.total_lab=total_lab
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self,idx):
        return self.total_data[idx],self.total_mask[idx],self.total_lab[idx]



def train_data_tox():
    data_frame={}

    tox=pd.read_csv('./origin/tox21.csv',header=0)
    # tox=tox.applymap(lambda x: x if str(x) != 'nan' else 2.0)
    tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    tox=tox.dropna()
    smiles=tokenizer(list(tox['smiles'].values),truncation=True,padding="max_length",max_length=512,return_tensors="pt")
    for name in list(tox.columns)[:-2]:
        data_frame[name]={'train':dataset(smiles['input_ids'].long()[:int(len(tox)*0.8)],smiles['attention_mask'].long()[:int(len(tox)*0.8)],torch.tensor(list(tox[name].values)).long()[:int(len(tox)*0.8)]),'dev':dataset(smiles['input_ids'].long()[int(len(tox)*0.8):int(len(tox)*0.9)],smiles['attention_mask'].long()[int(len(tox)*0.8):int(len(tox)*0.9)],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.8):int(len(tox)*0.9)]),'test':dataset(smiles['input_ids'].long()[int(len(tox)*0.9):],smiles['attention_mask'].long()[int(len(tox)*0.9):],torch.tensor(list(tox[name].values)).long()[int(len(tox)*0.9):])}

    return data_frame


def train_data_rf():
    from rdkit.Chem import Descriptors, MolFromSmiles
    from rdkit.ML.Descriptors import MoleculeDescriptors
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)

    # smiles = 'CC(=O)Nc1ccc2c(c1)C(=O)N(C(=O)N2C)C'
    # mol = MolFromSmiles(smiles)
    # descs = desc_calc.CalcDescriptors(mol)

    data_frame={}
    f=open(r'C:\Users\83912\Desktop\project\chemBert\data\data.json','r',encoding='utf-8')
    data=json.load(f)
    data_tmp={}

    f=open(r'C:\Users\83912\Desktop\project\chemBert\data\descs.json','w',encoding='utf-8')


    i=0
    for smile in data:
        i+=1
        if i>110000:
            break
        for t in data_title:
            if t not in data[smile]:
                continue
            if t not in data_tmp:
                data_tmp[t]={}
            mol=MolFromSmiles(smile)
            descs=','.join([str(v) for v in desc_calc.CalcDescriptors(mol)])
            data_tmp[t][descs]=data[smile][t]
    json.dump(data_tmp,f,ensure_ascii=False)
    train_num=1000

    for t in data_tmp:
        input_ids=list(data_tmp[t].keys())
        input_ids=[[float(v) for v in v.split(',')] for v in input_ids]
        value=[float(v) for v in list(data_tmp[t].values())]

        data_frame[t]={'train':[input_ids[:train_num],value[:train_num]],'dev':[input_ids[train_num:int(train_num*1.02)],value[train_num:int(train_num*1.02)]],'test':[input_ids[int(train_num*1.02):int(train_num*1.04)],value[int(train_num*1.02):int(train_num*1.04)]]}

    # print(torch.tensor([float(v) for v in list(data_tmp[t].values())]))
    return data_frame

def train_data_bert(data_path,tokenizer):
    data_frame={}
    f=open(data_path,'r',encoding='utf-8')
    data=json.load(f)
    data_tmp={}
    i=0
    for smile in data:
        i+=1
        if i>5000:
            break
        for t in data_title:
            if t not in data[smile]:
                continue
            if t not in data_tmp:
                data_tmp[t]={}
            data_tmp[t][smile]=data[smile][t]
    train_num=400

    for t in data_tmp:
        smiles=tokenizer(list(data_tmp[t].keys()),truncation=True,padding="max_length",max_length=512,return_tensors="pt")
        # data_frame[t]={'train':dataset(smiles['input_ids'].long(),smiles['attention_mask'].long(),torch.tensor([float(v) for v in list(data_tmp[t].values())]).float())}
        data_frame[t]={'train':dataset(smiles['input_ids'].long()[:train_num],smiles['attention_mask'].long()[:train_num],torch.tensor([float(v) for v in list(data_tmp[t].values())]).float()[:train_num]),
                        'dev':dataset(smiles['input_ids'].long()[train_num:int(train_num*1.1)],smiles['attention_mask'].long()[train_num:int(train_num*1.1)],torch.tensor([float(v) for v in list(data_tmp[t].values())]).float()[train_num:int(train_num*1.1)]),
                        'test':dataset(smiles['input_ids'].long()[int(train_num*1.1):int(train_num*1.2)],smiles['attention_mask'].long()[int(train_num*1.1):int(int(train_num*1.2))],torch.tensor([float(v) for v in list(data_tmp[t].values())]).float()[int(train_num*1.1):int(train_num*1.2)])}

    # print(torch.tensor([float(v) for v in list(data_tmp[t].values())]))
    return data_frame

def random_mask(ids):
    for i in range(len(ids)):
        if ids[i] in [0,1,2,4]:
            continue
        if torch.rand(1)>0.85:
            if torch.rand(1)>0.2:
                ids[i]=4
            else:
                ids[i]=random.randint(5,9999)
    return ids


def pretrain_data_manage():
    pretarin_data={}
    f=open('.\data\smiles.txt','r',encoding='utf-8')
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

    pretarin_data['train']=dataset(torch.tensor([random_mask(s.ids) for s in smiles[:int(len(smiles)*0.9)]]).long(),torch.tensor([s.attention_mask for s in smiles[:int(len(smiles)*0.9)]]).long(),torch.tensor([s.ids for s in smiles[:int(len(smiles)*0.9)]])).long()
    pretarin_data['dev']=dataset(torch.tensor([random_mask(s.ids) for s in smiles[int(len(smiles)*0.9):int(len(smiles)*0.95)]]).long(),torch.tensor([s.attention_mask for s in smiles[int(len(smiles)*0.9):int(len(smiles)*0.95)]]).long(),torch.tensor([s.ids for s in smiles[int(len(smiles)*0.9):int(len(smiles)*0.95)]])).long()
    pretarin_data['test']=dataset(torch.tensor([random_mask(s.ids) for s in smiles[int(len(smiles)*0.95):]]).long(),torch.tensor([s.attention_mask for s in smiles[int(len(smiles)*0.95):]]).long(),torch.tensor([s.ids for s in smiles[int(len(smiles)*0.95):]])).long()

    return pretarin_data
    # print(pretarin_data['train'].total_data[0])
    






# from transformers import AutoTokenizer, AutoModel

# model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM",num_labels=2)


# print(data_frame['NR-AR']['train'].total_data[0])

# print(model(data_frame['NR-AR']['train'].total_data[3].unsqueeze(0),attention_mask=data_frame['NR-AR']['train'].total_mask[3].unsqueeze(0)).last_hidden_state.shape)

if __name__=='__main__':
    # train_data_bert()
    print(train_data_rf()[data_choose]['dev'][0])

    pass
    






