import torch
# import numpy as np
# from model import BertModel
# from transformers import BertConfig,AutoModelForCausalLM


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



# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem
# rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]-[C!H0:4]>>[C:1](=[O:2])-[O:3]-[C:4]')
# reactants = (AllChem.MolFromSmiles('O=C(O)CCCCC(=O)OCCCO'), AllChem.MolFromSmiles('O=C(O)CCCCC(=O)OCCCO'))
# products = rxn.RunReactants(reactants)

# # print(len(products))
# # for p in products:
# #     smi = AllChem.MolToSmiles(p[0])
# #     print(smi)

# mol = Chem.MolFromSmiles('OCC1CCC(CC1)COC(=O)c1cccc(c1)C(=O)OCC1CCC(CC1)COC(=O)c1cccc(c1)C(=O)OCC1CCC(CC1)CO')
# Draw.MolToImage(mol, size=(300,300), kekulize=True)
# Draw.ShowMol(mol, size=(300,300), kekulize=False)


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


# import matplotlib.pyplot as plt
# import numpy as np

# x=np.array(['50','100','200','500','1000','2000'])
# fig=plt.figure(figsize=(4, 4), dpi=200)

# res1=[64.9,76.7,81.6,83.2,85.1,85.7]
# res2=[61.2,75.5,82.3,84.1,85.3,86.0]
# res3=[63.4,77.9,81.2,82.8,84.6,85.5]

# res4=[56.3,65.2,70.5,74.9,77.3,78.8]
# res5=[53.7,62.1,68.7,72.4,75.9,77.2]
# res6=[57.1,66.8,71.1,73.6,76.3,77.7]

# plt.plot(x,res1,lw=0.5,ls='-',c='k',label='ChemBert溶解度')
# plt.plot(x,res2,lw=0.5,ls='-',c='b',label='ChemBert疏水性参数')
# plt.plot(x,res3,lw=0.5,ls='-',c='r',label='ChemBert极性表面积')

# plt.plot(x,res4,lw=0.5,ls='dashdot',c='k',label='ChemBerta溶解度')
# plt.plot(x,res5,lw=0.5,ls='dashdot',c='b',label='ChemBerta疏水性参数')
# plt.plot(x,res6,lw=0.5,ls='dashdot',c='r',label='ChemBerta极性表面积')
# plt.legend(fontsize=5,ncol=2)
# plt.show()
