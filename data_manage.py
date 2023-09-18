import os
import json
import pandas as pd
import math
smiles=['> <PUBCHEM_OPENEYE_CAN_SMILES>']  #'> <PUBCHEM_OPENEYE_ISO_SMILES>' 
names=['> <PUBCHEM_IUPAC_OPENEYE_NAME>' , '> <PUBCHEM_IUPAC_CAS_NAME>' , '> <PUBCHEM_IUPAC_NAME_MARKUP>' , '> <PUBCHEM_IUPAC_NAME>' , '> <PUBCHEM_IUPAC_SYSTEMATIC_NAME>' , '> <PUBCHEM_IUPAC_TRADITIONAL_NAME>']
data_title=['> <PUBCHEM_XLOGP3_AA>','> <PUBCHEM_EXACT_MASS>','> <PUBCHEM_MOLECULAR_WEIGHT>','> <PUBCHEM_CACTVS_TPSA>','> <PUBCHEM_MONOISOTOPIC_WEIGHT>','> <PUBCHEM_TOTAL_CHARGE>','> <PUBCHEM_HEAVY_ATOM_COUNT>']

# f=open('/mnt/nas-search-nlp/xzh/test/pubchem/Compound_000000001_000500000.sdf','r',encoding='utf-8')


def train_data():
    fdata=open('data.json','w',encoding='utf-8')

    files= os.listdir(r'C:\Users\83912\Desktop\project\chemBert\data')
    # files=[1]

    smiles_list=[]
    data_json={}
    data_total={}
    for file in files:

        file_path=r'C:\Users\83912\Desktop\project\chemBert\data\\'+file
        # file_path='C:\Users\83912\Desktop\project\chemRoBerta\data\tmp.txt'

        data=[]
        with open(file_path,'r',encoding='utf-8') as f:
            pubchem=f.readlines()
            for chem in pubchem:
                chem=chem.replace('\n','')
                if not chem:
                    continue
                if chem=='$$$$':
                    for s in smiles:
                        try:
                            smiles_list.append(data[data.index(s)+1])
                        except:
                            pass
                    for d in data_title:
                        try:
                            data_json[d]=data[data.index(d)+1]
                        except:
                            pass
                    
                    for s in smiles_list:
                        data_total[s]={}
                        for d in data_json:
                            data_total[s][d]=data_json[d]

                    smiles_list=[]
                    data_json={}
                    data=[]
                    pass

                data.append(chem)
    print(len(data_total))
    json.dump(data_total,fdata,ensure_ascii=False,indent=4)


def smiles_data(begin=0):
    f1=open('smiles.txt','w',encoding='utf-8')
    f2=open('names.txt','w',encoding='utf-8')

    files= os.listdir(r'C:\Users\83912\Desktop\project\chemBert\data')
    # files=[1]

    print(files[begin:])
    for file in files[begin:]:

        file_path=r'C:\Users\83912\Desktop\project\chemBert\data\\'+file

        data=[]
        with open(file_path,'r',encoding='utf-8') as f:
            pubchem=f.readlines()
            for chem in pubchem:
                chem=chem.replace('\n','')
                if not chem:
                    continue
                if chem=='$$$$':
                    for s in smiles:
                        try:
                            f1.write(data[data.index(s)+1]+'\n')
                        except:
                            pass
                    for n in names:
                        try:
                            f2.write(data[data.index(n)+1]+'\n')
                        except:
                            pass
                    data=[]
                    pass
                data.append(chem)


# train_data()
# smiles_data()

# df=pd.read_excel('/mnt/workspace/chemBert/data/51.xlsx')
# # f=open('/mnt/workspace/chemBert/data/51.json')
# data={}
# for line in pd.DataFrame(df).itertuples():
#     data[line._12]={'D_MS刚性-C3H8':line._15*1e9}
#     data[line._13]={'D_MS刚性-C3H8':line._15*1e9}
#     data[line._14]={'D_MS刚性-C3H8':line._15*1e9}
# json.dump(data,open('/mnt/workspace/chemBert/data/COF-smile.json','w'),ensure_ascii=False,indent=4)

df=pd.read_excel('./origin/crystal.xlsx')
data={}
for line in pd.DataFrame(df).itertuples():
    data[line.smiles]={'formation_energy':line.formation_energy,'band_gap':line.band_gap}
json.dump(data,open('./data/crystal.json','w'),ensure_ascii=False,indent=4)



# f=open('./data/solubility.txt')
# data={}
# res=0
# for line in f.readlines():
#     line=line.split()
#     data[line[0]]={'solubility':line[2]}
#     res+=math.exp(float(line[2]))
# json.dump(data,open('./data/solubility.json','w'),ensure_ascii=False,indent=4)
# print(res/len(data))
# print(data)

# print(data[data.index('> <PUBCHEM_OPENEYE_CAN_SMILES>')+1])


