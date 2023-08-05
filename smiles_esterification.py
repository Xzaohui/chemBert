from rdkit import Chem
from rdkit.Chem import AllChem
import json
from tqdm import tqdm
from rdkit.Chem import Descriptors

# 酯化 酸+醇
# rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]-[C!H0:4]>>[C:1](=[O:2])-[O:3]-[C:4]')
# reactants = (AllChem.MolFromSmiles('OCCO'), AllChem.MolFromSmiles('OCCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))


# 基团保护
# rxn = AllChem.ReactionFromSmarts('[B:1].[O!H0:2]>>[O:2]-[X:1]')
# reactants = (AllChem.MolFromSmiles('B'), AllChem.MolFromSmiles('BOC(CC(=O)O)(CC(=O)O)C(=O)O'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

# reactant_smiles = "BOC(=O)C(CC(=O)O)(CC(=O)O)OB"
# reactant = Chem.MolFromSmiles(reactant_smiles)
# break_b_o_smarts = "[B:1]-[O:2]"
# break_b_o_pattern = Chem.MolFromSmarts(break_b_o_smarts)
# matches = reactant.GetSubstructMatches(break_b_o_pattern)
# product = Chem.RWMol(reactant)
# for match in matches:
#     atom_b_index, atom_o_index = match
#     product.RemoveBond(atom_b_index, atom_o_index)
# product = product.GetMol()
# product_smiles = Chem.MolToSmiles(product)
# ans=product_smiles.split('.')[-1]

oh=['C(CO)O','C(COCCO)O','CC(CO)O','OCCCCO','CC(C)(CO)CO','CC(CO)CO','C(CCCO)CCO','CC(CCO)O','C(CO)CO','C(CCO)CCO','CC(CCO)CCO','CCC(CC(CC)CO)CO','CC(C)C(C(C)(C)CO)O','CCC(O)OC(CC)O','C1CC(CCC1CO)CO','C1CC(CCC1O)O','CCCCC(CC)(CO)CO','CCCC(C(CC)CO)O','C(CCCCCCO)CCCCCO','OC1CCCCCCCCC1O','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O','C[Si](C)(O)O','C1=CC=C(C=C1)[Si](C2=CC=CC=C2)(O)O','CC(CC(C(F)(F)F)(C(F)(F)F)O)O','C(C(C(C(C(CO)(F)F)(F)F)(F)F)(F)F)O']
cooh=['C(CCC(=O)O)CC(=O)O','C(CCCCC(=O)O)CCCC(=O)O','C1=CC(=CC=C1C(=O)O)C(=O)O','C1=CC=C2C(=C1)C(=O)OC2=O','C1=CC(=CC(=C1)C(=O)O)C(=O)O','C(CC(=O)O)C(=O)O','C(CC(=O)O)CC(=O)O','C(CCCC(=O)O)CCCC(=O)O','C(CCCCCC(=O)O)CCCCC(=O)O','C1CC(CCC1C(=O)O)C(=O)O','CCCCCCCCC(CCCCCCCCC(=O)O)C(CCCCCCCC(=O)O)CCC=CCCCCC','C=C(CC(=O)O)C(=O)O']
moh=['CCC(CO)(CO)CO','C(C(CO)O)O','C(C(CO)(CO)CO)O','CCCCCCC(CC=CCCCCCCCC(=O)OCC(COC(=O)CCCCCCCC=CCC(CCCCCC)O)OC(=O)CCCCCCCC=CCC(CCCCCC)O)O','C(C(C(C(C(C=O)O)O)O)O)O','CC(CCO)(CCO)O','C(CO)N(CCO)CCO']
mcooh=['C1=CC2=C(C=C1C(=O)O)C(=O)OC2=O','c1c(cc(cc1C(=O)O)C(=O)O)C(=O)O','C1=C(C=C(C=C1C(=O)O)C(=O)O)C(=O)O']
m=['CC(CO)(CO)C(=O)O','CCC(CO)(CO)C(=O)O','C(C(=O)O)C(CC(=O)O)(C(=O)O)O','CC(C)(CO)C(=O)O']

t_material=oh+cooh
m_material=moh+mcooh+m

intermediate_product={}
final_product={}

for i in range(len(t_material)):
    intermediate_product[t_material[i]]=[[t_material[i]],'常见']

for i in range(len(m_material)):
    functional_group = Chem.MolFromSmarts("[O!H0]")
    tmp1=[m_material[i]]
    while True:
        flag=0
        tmp2=[]
        for j in range(len(tmp1)):
            reactant = Chem.MolFromSmiles(tmp1[j])
            matches = len(reactant.GetSubstructMatches(functional_group))
            if matches<3:
                continue
            rxn = AllChem.ReactionFromSmarts('[B:1].[O!H0:2]>>[O:2]-[X:1]')
            reactants = (AllChem.MolFromSmiles('B'), AllChem.MolFromSmiles(tmp1[j]))
            products = rxn.RunReactants(reactants)
            for p in products:
                tmp2.append(AllChem.MolToSmiles(p[0]))
            flag=1
        if flag==0:
            break
        else:
            tmp1=tmp2
            
    for p in tmp1:
        intermediate_product[p]=[[m_material[i]],'常见']

tmp_product=intermediate_product.copy()
while True:
    flag=0
    for product1,v in intermediate_product.items():
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]-[C!H0:4]>>[C:1](=[O:2])-[O:3]-[C:4]')
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

for product1,v in tqdm(intermediate_product.items()):
    reactant = Chem.MolFromSmiles(product1)
    break_b_o_pattern = Chem.MolFromSmarts("[B:1]-[O:2]")
    matches = reactant.GetSubstructMatches(break_b_o_pattern)

    if len(matches) > 0:
        product = Chem.RWMol(reactant)
        for match in matches:
            atom_b_index, atom_o_index = match
            product.RemoveBond(atom_b_index, atom_o_index)
        product = product.GetMol()
        product_smiles = Chem.MolToSmiles(product)
        product1=product_smiles.split('.')[-1]
    
    reactant = Chem.MolFromSmiles(product1)
    functional_group = Chem.MolFromSmarts("[C:1](=[O:2])-[OD1]")
    matches = reactant.GetSubstructMatches(functional_group)
    if len(matches) > 0:
        l_oh=list(set(oh+moh)&set(v[0]))
        if len(l_oh)==0:
            continue

        tmp1=[product1]
        while True:
            flag=0
            tmp2=[]
            for j in range(len(tmp1)):
                reactant = Chem.MolFromSmiles(tmp1[j])
                matches = len(reactant.GetSubstructMatches(functional_group))
                
                if matches==0:
                    continue
                rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]-[C!H0:4]>>[C:1](=[O:2])-[O:3]-[C:4]')
                for product2 in l_oh:
                    reactants = (AllChem.MolFromSmiles(tmp1[j]), AllChem.MolFromSmiles(product2))
                    products = rxn.RunReactants(reactants)
                    for p in products:
                        tmp2.append(AllChem.MolToSmiles(p[0]))
                flag=1
            if flag==0:
                break
            else:
                tmp1=tmp2
        for p in tmp1:
            final_product[p]=v
    else:
        final_product[product1]=v

    

json.dump(final_product,open('smiles_final.json','w'),indent=4,ensure_ascii=False)



