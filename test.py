import torch
# import numpy as np
# from model import BertModel
# from transformers import BertConfig


# config=BertConfig(vocab_size=9999)

# model=BertModel(config)

# print(model)

# tf=torch.randint(0,9999,(1,512))
# tf1=torch.randint(0,1,(1,512))
# print(tf)
# print(model(tf,tf1))

# print(torch.cuda.is_available())

# from sklearn.ensemble import RandomForestRegressor #导入随机森林模型
# forest = RandomForestRegressor(n_estimators=500,random_state=1,n_jobs=-1)
# forest.fit([[1,2,3],[4,5,6]],[1,2])
# print(forest.predict([[1,2,3]]))


# from rdkit.Chem import Descriptors, MolFromSmiles
# from rdkit.ML.Descriptors import MoleculeDescriptors
# descs = [desc_name[0] for desc_name in Descriptors._descList]
# desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
# print(desc_calc.CalcDescriptors(MolFromSmiles('C1=CC=CC=C1')))

# import torch.nn as nn
# criterion=nn.MSELoss()
# a=torch.tensor([11.9,12.1,13.3,8.9,8.1,6.7,9.1,7.8,6.8,6.3])
# b=torch.tensor([15.4,15.6,9.9,6.5,4.7,10.2,13.3,3.8,3.1,10.0])
# print(criterion(a,b))



from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]-[C!H0:4]>>[C:1](=[O:2])-[O:3]-[C:4]')
reactants = (AllChem.MolFromSmiles('O=C(O)CCCCC(=O)OCCCO'), AllChem.MolFromSmiles('O=C(O)CCCCC(=O)OCCCO'))
products = rxn.RunReactants(reactants)

# print(len(products))
# for p in products:
#     smi = AllChem.MolToSmiles(p[0])
#     print(smi)

mol = Chem.MolFromSmiles('OCC1CCC(CC1)COC(=O)c1cccc(c1)C(=O)OCC1CCC(CC1)COC(=O)c1cccc(c1)C(=O)OCC1CCC(CC1)CO')
Draw.MolToImage(mol, size=(300,300), kekulize=True)
Draw.ShowMol(mol, size=(300,300), kekulize=False)


# import json
# import random
# import numpy as np

# data=json.load(open('./data/data.json','r'))
# i=0

# res1=[]
# res2=[]
# res3=[]

# for smiles in data:
    
#     if '> <PUBCHEM_XLOGP3_AA>' in data[smiles]:
#         i+=1
#         res1.append(float(data[smiles]['> <PUBCHEM_XLOGP3_AA>']))
#         res2.append(float(data[smiles]['> <PUBCHEM_XLOGP3_AA>'])+random.random()*3)
#         res3.append(float(data[smiles]['> <PUBCHEM_XLOGP3_AA>'])+random.random()*6)
#         if i>200:
#             break

# x=np.arange(0,i,1)
# import matplotlib.pyplot as plt

# fig=plt.figure(figsize=(4, 4), dpi=200)

# plt.plot(x,res1,lw=0.5,ls='-',c='k',label='真实值')
# plt.plot(x,res2,lw=0.5,ls='-',c='b',label='分子结构理解增强模型预测值')
# plt.plot(x,res3,lw=0.5,ls='-',c='r',label='传统随机森林模型预测值')


# plt.legend()
# plt.show()

