from rdkit import Chem
from rdkit.Chem import AllChem
import json
from tqdm import tqdm
from rdkit.Chem import Descriptors

oh=['C(CO)O','C(COCCO)O','CC(CO)O','OCCCCO','CC(C)(CO)CO','CC(CO)CO','C(CCCO)CCO','CC(CCO)O','C(CO)CO','C(CCO)CCO','CC(CCO)CCO','CCC(CC(CC)CO)CO','CC(C)C(C(C)(C)CO)O','CCC(O)OC(CC)O','C1CC(CCC1CO)CO','C1CC(CCC1O)O','CCCCC(CC)(CO)CO','CCCC(C(CC)CO)O','C(CCCCCCO)CCCCCO','OC1CCCCCCCCC1O','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O','C[Si](C)(O)O','C1=CC=C(C=C1)[Si](C2=CC=CC=C2)(O)O','CC(CC(C(F)(F)F)(C(F)(F)F)O)O','C(C(C(C(C(CO)(F)F)(F)F)(F)F)(F)F)O']


# 聚醚
# rxn = AllChem.ReactionFromSmarts('[C:1]-[O!H0:2].[O!H0:3]-[C!H0:4]>>[C:1]-[O:3]-[C:4]')
# reactants = (AllChem.MolFromSmiles('OCCO'), AllChem.MolFromSmiles('OCCC(C)CCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

intermediate_product={}

for i in range(len(oh)):
    intermediate_product[oh[i]]=[[oh[i]],'常见']


tmp_product=intermediate_product.copy()
while True:
    flag=0
    for product1,v in intermediate_product.items():
        rxn = AllChem.ReactionFromSmarts('[C:1]-[O!H0:2].[O!H0:3]-[C!H0:4]>>[C:1]-[O:3]-[C:4]')
        for product2 in intermediate_product.keys():
            mol1 = Chem.MolFromSmiles(product1)
            mol2 = Chem.MolFromSmiles(product2)
            if len(set(intermediate_product[product1][0]+intermediate_product[product2][0]))>2 or Descriptors.MolWt(mol1)+Descriptors.MolWt(mol2)>5000:
                continue
            reactants = (mol1, mol2)
            products = rxn.RunReactants(reactants)
            for p in products:
                if AllChem.MolToSmiles(p[0]) not in tmp_product.keys():
                    tmp_product[AllChem.MolToSmiles(p[0])]=[list(set(v[0]+intermediate_product[product2][0])),'常见']
                    flag=1
    intermediate_product=tmp_product.copy()
    if len(intermediate_product)>1e3:
        break
    if flag==0:
        break

json.dump(intermediate_product, open('smiles_polyether_final.json','w'),indent=4,ensure_ascii=False)