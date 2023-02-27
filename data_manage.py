import os
smiles=['> <PUBCHEM_OPENEYE_CAN_SMILES>' , '> <PUBCHEM_OPENEYE_ISO_SMILES>' ]
names=['> <PUBCHEM_IUPAC_OPENEYE_NAME>' , '> <PUBCHEM_IUPAC_CAS_NAME>' , '> <PUBCHEM_IUPAC_NAME_MARKUP>' , '> <PUBCHEM_IUPAC_NAME>' , '> <PUBCHEM_IUPAC_SYSTEMATIC_NAME>' , '> <PUBCHEM_IUPAC_TRADITIONAL_NAME>']

f=open('/mnt/nas-search-nlp/xzh/test/pubchem/Compound_000000001_000500000.sdf','r',encoding='utf-8')
f1=open('smiles.txt','w',encoding='utf-8')
f2=open('names.txt','w',encoding='utf-8')



files= os.listdir('/mnt/nas-search-nlp/xzh/test/pubchem')

for file in files:
    file_path='/mnt/nas-search-nlp/xzh/test/pubchem/'+file
    data=[]
    with open(file_path,'r',encoding='utf-8') as f:
        pubchem=f.readlines()
        for chem in pubchem:
            chem=chem.replace('\n','')
            if chem=='$$$$':
                for s in smiles:
                    try:
                        f1.write(data[data.index(s)+1]+'\n\n')
                    except:
                        pass
                for n in names:
                    try:
                        f2.write(data[data.index(n)+1]+'\n\n')
                    except:
                        pass
                data=[]
                pass
            data.append(chem)


# for line in f.readlines():
#     line=line.strip('\n')
#     if line=='$$$$':
#         break
#     data.append(line)

# print(data)

# print(data[data.index('> <PUBCHEM_OPENEYE_CAN_SMILES>')+1])


