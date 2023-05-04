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


from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.ML.Descriptors import MoleculeDescriptors
descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
print(desc_calc.CalcDescriptors(MolFromSmiles('C1=CC=CC=C1')))

