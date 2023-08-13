from rdkit import Chem
from rdkit.Chem import AllChem
import json
from tqdm import tqdm
from rdkit.Chem import Descriptors

epoxy=['CC(C)(COCC1CO1)COCC2CO2','C1C(O1)COCCOCC2CO2','C1C(O1)COCC(COCC2CO2)O','C[Si](C)(CCCOCC1CO1)O[Si](C)(C)O[Si](C)(C)CCCOCC2CO2','C1C(O1)COCCCCOCC2CO2','CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4','C1CCC(C(C1)OCC2CO2)OCC3CO3','CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O.C1C(O1)CCl.C(CO)N','CCCCCC1C(O1)CC2C(O2)CCCCCCCC(=O)OCC(COC(=O)CCCCCCCC3C(O3)CC4C(O4)CCCCC)OC(=O)CCCCCCCC5C(O5)CC6C(O6)CCCCC','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O.CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4','CC(=C)C(=O)OCC1CO1','CCCCCC1C(O1)CC2C(O2)CCCCCCCC(=O)OCC(COC(=O)CCCCCCCC3C(O3)CC4C(O4)CCCCC)OC(=O)CCCCCCCC5C(O5)CC6C(O6)CCCCC']
oh=['C(CO)O','C(COCCO)O','CC(CO)O','OCCCCO','CC(C)(CO)CO','CC(CO)CO','C(CCCO)CCO','CC(CCO)O','C(CO)CO','C(CCO)CCO','CC(CCO)CCO','CCC(CC(CC)CO)CO','CC(C)C(C(C)(C)CO)O','CCC(O)OC(CC)O','C1CC(CCC1CO)CO','C1CC(CCC1O)O','CCCCC(CC)(CO)CO','CCCC(C(CC)CO)O','C(CCCCCCO)CCCCCO','OC1CCCCCCCCC1O','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O','C[Si](C)(O)O','C1=CC=C(C=C1)[Si](C2=CC=CC=C2)(O)O','CC(CC(C(F)(F)F)(C(F)(F)F)O)O','C(C(C(C(C(CO)(F)F)(F)F)(F)F)(F)F)O']
cooh=['C(CCC(=O)O)CC(=O)O','C(CCCCC(=O)O)CCCC(=O)O','C1=CC(=CC=C1C(=O)O)C(=O)O','C1=CC=C2C(=C1)C(=O)OC2=O','C1=CC(=CC(=C1)C(=O)O)C(=O)O','C(CC(=O)O)C(=O)O','C(CC(=O)O)CC(=O)O','C(CCCC(=O)O)CCCC(=O)O','C(CCCCCC(=O)O)CCCCC(=O)O','C1CC(CCC1C(=O)O)C(=O)O','CCCCCCCCC(CCCCCCCCC(=O)O)C(CCCCCCCC(=O)O)CCC=CCCCCC','C=C(CC(=O)O)C(=O)O']
ooh=['CCCCCCCCCCCCCCCC1=CC(=CC=C1)O','COC1=C(C=CC(=C1)C=CCO)O','COC1=CC(=CC(=C1O)OC)C=CCO','C1=CC(=CC=C1C=CCO)O','C(CO)C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F','C=CC(=O)OCCO','CC(=C)C(=O)OCCO','C=CC(=O)OCC(CO)(COC(=O)C=C)COC(=O)C=C']
ocooh=['CCCCCCC(CC=CCCCCCCCC(=O)O)O','CCCCCC=CCC=CCCCCCCCC(=O)O']
moh=['CCC(CO)(CO)CO','C(C(CO)O)O','C(C(CO)(CO)CO)O','CCCCCCC(CC=CCCCCCCCC(=O)OCC(COC(=O)CCCCCCCC=CCC(CCCCCC)O)OC(=O)CCCCCCCC=CCC(CCCCCC)O)O','C(C(C(C(C(C=O)O)O)O)O)O','CC(CCO)(CCO)O','C(CO)N(CCO)CCO']
mcooh=['C1=CC2=C(C=C1C(=O)O)C(=O)OC2=O','c1c(cc(cc1C(=O)O)C(=O)O)C(=O)O','C1=C(C=C(C=C1C(=O)O)C(=O)O)C(=O)O']
m=['CC(CO)(CO)C(=O)O','CCC(CO)(CO)C(=O)O','C(C(=O)O)C(CC(=O)O)(C(=O)O)O','CC(C)(CO)C(=O)O']

# 环氧开环 
# 醇
# rxn = AllChem.ReactionFromSmarts('[C:1]-[C:2]1-[CH2:3]-[O:4]1.[O:5]-[C:6]>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:5]-[C:6]')
# reactants = (AllChem.MolFromSmiles('CCOCC(O)COCC(C)(C)COCC1CO1'), AllChem.MolFromSmiles('CCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

# 酸
# rxn = AllChem.ReactionFromSmarts('[C:1]-[C:2]1-[CH2:3]-[O:4]1.[C:5](=[O:6])-[OH:7]>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:7]-[C:5](=[O:6])')
# reactants = (AllChem.MolFromSmiles('CC(C)(COCC(O)COC(=O)CCC(=O)O)COCC1CO1'), AllChem.MolFromSmiles('O=C(O)CCCCC(=O)O'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))


simple=oh+ooh+ocooh+moh

tmp_product={}
intermediate_product={}
final_product={}

for i in range(len(epoxy)):
    intermediate_product[epoxy[i]]=[[epoxy[i]],'常见','结构单元']


for product1,v in intermediate_product.items():
    rxn = AllChem.ReactionFromSmarts('[C:1]-[C:2]1-[CH2:3]-[O:4]1.[O:5]-[C:6]>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:5]-[C:6]')
    for product2 in simple:
        mol1 = Chem.MolFromSmiles(product1)
        mol2 = Chem.MolFromSmiles(product2)
        reactants = (mol1, mol2)
        products = rxn.RunReactants(reactants)
        for p in products:
            if AllChem.MolToSmiles(p[0]) not in tmp_product.keys():
                tmp_product[AllChem.MolToSmiles(p[0])]=[list(set(v[0]+[product2])),'常见','结构单元']
intermediate_product=tmp_product.copy()
tmp_product={}
for product1,v in intermediate_product.items():
    rxn = AllChem.ReactionFromSmarts('[C:1]-[C:2]1-[CH2:3]-[O:4]1.[O:5]-[C:6]>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:5]-[C:6]')
    for product2 in simple:
        mol1 = Chem.MolFromSmiles(product1)
        mol2 = Chem.MolFromSmiles(product2)
        reactants = (mol1, mol2)
        try:
            products = rxn.RunReactants(reactants)
            for p in products:
                if AllChem.MolToSmiles(p[0]) not in tmp_product.keys():
                    tmp_product[AllChem.MolToSmiles(p[0])]=[list(set(v[0]+[product2])),'常见','结构单体']
        except:
            pass
intermediate_product=tmp_product.copy()



json.dump(intermediate_product, open('smiles_epoxy_final.json','w'),indent=4,ensure_ascii=False)