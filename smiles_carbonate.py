from rdkit import Chem
from rdkit.Chem import AllChem
import json
from tqdm import tqdm
from rdkit.Chem import Descriptors

coc="COC(=O)OC"
oh=['C(CO)O','C(COCCO)O','CC(CO)O','OCCCCO','CC(C)(CO)CO','CC(CO)CO','C(CCCO)CCO','CC(CCO)O','C(CO)CO','C(CCO)CCO','CC(CCO)CCO','CCC(CC(CC)CO)CO','CC(C)C(C(C)(C)CO)O','CCC(O)OC(CC)O','C1CC(CCC1CO)CO','C1CC(CCC1O)O','CCCCC(CC)(CO)CO','CCCC(C(CC)CO)O','C(CCCCCCO)CCCCCO','OC1CCCCCCCCC1O','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O','C[Si](C)(O)O','C1=CC=C(C=C1)[Si](C2=CC=CC=C2)(O)O','CC(CC(C(F)(F)F)(C(F)(F)F)O)O','C(C(C(C(C(CO)(F)F)(F)F)(F)F)(F)F)O']

# 碳酸酯
# rxn = AllChem.ReactionFromSmarts('[C:1]-[O!H0:2].[CH3:3]-[O:4]-[C:5](=[O:6])>>[C:1]-[O:4]-[C:5](=[O:6])')
# reactants = (AllChem.MolFromSmiles('OCCO'), AllChem.MolFromSmiles('COC(=O)OCCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

intermediate_product={}

intermediate_product[coc]=[[coc],'常见']


tmp_product={}

for product1,v in intermediate_product.items():
    rxn = AllChem.ReactionFromSmarts('[C:1]-[O!H0:2].[CH3:3]-[O:4]-[C:5](=[O:6])>>[C:1]-[O:4]-[C:5](=[O:6])')
    for product2 in oh:
        mol1 = Chem.MolFromSmiles(product2)
        mol2 = Chem.MolFromSmiles(product1)
        reactants = (mol1, mol2)
        products = rxn.RunReactants(reactants)
        for p in products:
            if AllChem.MolToSmiles(p[0]) not in tmp_product.keys():
                tmp_product[AllChem.MolToSmiles(p[0])]=[list(set(v[0]+[product2])),'常见']
intermediate_product=tmp_product.copy()
tmp_product={}
for product1,v in intermediate_product.items():
    rxn = AllChem.ReactionFromSmarts('[C:1]-[O!H0:2].[CH3:3]-[O:4]-[C:5](=[O:6])>>[C:1]-[O:4]-[C:5](=[O:6])')
    for product2 in oh:
        mol1 = Chem.MolFromSmiles(product2)
        mol2 = Chem.MolFromSmiles(product1)
        reactants = (mol1, mol2)
        products = rxn.RunReactants(reactants)
        for p in products:
            if AllChem.MolToSmiles(p[0]) not in tmp_product.keys():
                tmp_product[AllChem.MolToSmiles(p[0])]=[list(set(v[0]+[product2])),'常见']
intermediate_product=tmp_product.copy()



json.dump(intermediate_product, open('smiles_carbonate_final.json','w'),indent=4,ensure_ascii=False)