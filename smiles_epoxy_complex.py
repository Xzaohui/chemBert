from rdkit import Chem
from rdkit.Chem import AllChem
import json
from tqdm import tqdm
from rdkit.Chem import Descriptors

epoxy=['CC(C)(COCC1CO1)COCC2CO2','C1C(O1)COCCOCC2CO2','C1C(O1)COCC(COCC2CO2)O','C[Si](C)(CCCOCC1CO1)O[Si](C)(C)O[Si](C)(C)CCCOCC2CO2','C1C(O1)COCCCCOCC2CO2','CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4','C1CCC(C(C1)OCC2CO2)OCC3CO3','CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O.C1C(O1)CCl.C(CO)N','CCCCCC1C(O1)CC2C(O2)CCCCCCCC(=O)OCC(COC(=O)CCCCCCCC3C(O3)CC4C(O4)CCCCC)OC(=O)CCCCCCCC5C(O5)CC6C(O6)CCCCC','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O.CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4','CC(=C)C(=O)OCC1CO1','CCCCCC1C(O1)CC2C(O2)CCCCCCCC(=O)OCC(COC(=O)CCCCCCCC3C(O3)CC4C(O4)CCCCC)OC(=O)CCCCCCCC5C(O5)CC6C(O6)CCCCC']

carbonate=json.load(open('smiles_carbonate_final.json','r'))
polyether=json.load(open('smiles_polyether_tmp.json','r'))
esterification=json.load(open('smiles_esterification_tmp.json','r'))

tmp_product={}
intermediate_product={}
final_product={}

for i in range(len(epoxy)):
    intermediate_product[epoxy[i]]=[[epoxy[i]],'常见','结构单元']


for product1,v in carbonate.items():
    rxn = AllChem.ReactionFromSmarts('[O:5]-[C:6].[C:1]-[C:2]1-[CH2:3]-[O:4]1>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:5]-[C:6]')
    for product2 in epoxy:
        mol1 = Chem.MolFromSmiles(product1)
        mol2 = Chem.MolFromSmiles(product2)
        reactants = (mol1, mol2)
        products = rxn.RunReactants(reactants)
        for p in products:
            if AllChem.MolToSmiles(p[0]) not in tmp_product.keys():
                tmp_product[AllChem.MolToSmiles(p[0])]=[list(set(v[0]+[product2])),'常见','结构单元']


for product1,v in polyether.items():
    rxn = AllChem.ReactionFromSmarts('[O:5]-[C:6].[C:1]-[C:2]1-[CH2:3]-[O:4]1>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:5]-[C:6]')
    for product2 in epoxy:
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

for product1,v in esterification.items():
    rxn = AllChem.ReactionFromSmarts('[O:5]-[C:6].[C:1]-[C:2]1-[CH2:3]-[O:4]1>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:5]-[C:6]')
    for product2 in epoxy:
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


json.dump(tmp_product, open('smiles_epoxy_complex_final.json','w'),indent=4,ensure_ascii=False)