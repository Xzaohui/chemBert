# oh=['CCC(C)CC','C1CCC(CC1)','CC1CCC(CC1)C','CC','C(C)C']
# cooh=['CCCC','CCCCCCCC','c1cccc(c1)','c1ccccc1','c1ccc(cc1)']


# for i in range(len(oh)):
#     oh[i]='O'+oh[i]+'O'

# for i in range(len(cooh)):
#     cooh[i]='O=C(O)'+cooh[i]+'C(=O)O'

# toh=oh[:]
# tcooh=cooh[:]


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# 聚醚
# rxn = AllChem.ReactionFromSmarts('[C:1]-[O!H0:2].[O!H0:3]-[C!H0:4]>>[C:1]-[O:3]-[C:4]')
# reactants = (AllChem.MolFromSmiles('OCCO'), AllChem.MolFromSmiles('OCCC(C)CCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

# 碳酸酯
# rxn = AllChem.ReactionFromSmarts('[C:1]-[O!H0:2].[CH3:3]-[O:4]-[C:5](=[O:6])>>[C:1]-[O:4]-[C:5](=[O:6])')
# reactants = (AllChem.MolFromSmiles('OCCO'), AllChem.MolFromSmiles('COC(=O)OCCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

# 基团保护
# rxn = AllChem.ReactionFromSmarts('[B:1].[O!H0:2]>>[O:2]-[X:1]')
# reactants = (AllChem.MolFromSmiles('B'), AllChem.MolFromSmiles('BOC(CC(=O)O)(CC(=O)O)C(=O)O'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

# 聚酯 酸+醇
# rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]-[C!H0:4]>>[C:1](=[O:2])-[O:3]-[C:4]')
# reactants = (AllChem.MolFromSmiles('OCCO'), AllChem.MolFromSmiles('OCCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))


# 环氧开环 
# 醇
# rxn = AllChem.ReactionFromSmarts('[C:1]-[C:2]1-[CH2:3]-[O:4]1.[O:5]-[C:6]>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:5]-[C:6]')
# reactants = (AllChem.MolFromSmiles('CC(C)(COCC(O)CO(CCO)CCO)COCC1CO1'), AllChem.MolFromSmiles('OCCC(C)CCO'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))

# 酸
# rxn = AllChem.ReactionFromSmarts('[C:1]-[C:2]1-[CH2:3]-[O:4]1.[C:5](=[O:6])-[OH:7]>>[C:1]-[C:2](-[OH:4])-[C:3]-[O:7]-[C:5](=[O:6])')
# reactants = (AllChem.MolFromSmiles('CC(C)(COCC(O)COC(=O)CCC(=O)O)COCC1CO1'), AllChem.MolFromSmiles('O=C(O)CCCCC(=O)O'))
# products = rxn.RunReactants(reactants)
# for p in products:
#     print(AllChem.MolToSmiles(p[0]))


# res=set()
# for i in cooh:
#     for j in oh:
#         reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
#         products = rxn.RunReactants(reactants)
#         for p in products:
#             smi = AllChem.MolToSmiles(p[0])
#             res.add(smi)
#             break

# oh+=list(res)
# cooh+=list(res)

# # print(list(res))

# res=set()
# for i in cooh:
#     for j in oh:
#         reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
#         products = rxn.RunReactants(reactants)
#         for p in products:
#             smi = AllChem.MolToSmiles(p[0])
#             res.add(smi)
#             break



# oh+=list(res)
# oh=list(set(oh))
# cooh+=list(res)
# cooh=list(set(cooh))

# # print(list(res))

# t=list(res)
# res=set()
# for i in t:
#     for j in toh:
#         reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
#         products = rxn.RunReactants(reactants)
#         for p in products:
#             smi = AllChem.MolToSmiles(p[0])
#             res.add(smi)
#             break

# t=list(res)
# res=set()
# for i in t:
#     for j in toh:
#         reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
#         products = rxn.RunReactants(reactants)
#         for p in products:
#             smi = AllChem.MolToSmiles(p[0])
#             res.add(smi)
#             break

# print(list(res))


# res=set()
# for i in cooh:
#     for j in oh:
#         reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
#         products = rxn.RunReactants(reactants)
#         for p in products:
#             smi = AllChem.MolToSmiles(p[0])
#             res.add(smi)

# print(list(res))

mol = Chem.MolFromSmiles('CC(C)(COCC(O)COCCOC(=O)OCCO)COCC1CO1')

Draw.MolToImage(mol, size=(300,300), kekulize=True)
Draw.ShowMol(mol, size=(300,300), kekulize=False)

# from rdkit.Chem import Descriptors
# print(Descriptors.MolWt(mol)) # 计算分子量

# from rdkit import Chem
# from rdkit.Chem import Draw
# m = Chem.MolFromSmiles('OC(=O)CC(O)(CC(=O)OCC)C(=O)O')
# p = Chem.MolFromSmarts('[OH]C(=O)CC(O)(C(=O)O)CC(=O)O')
# print(m.GetSubstructMatch(p))


# oh=['C(CO)O','C(COCCO)O','CC(CO)O','OCCCCO','CC(C)(CO)CO','CC(CO)CO','C(CCCO)CCO','CC(CCO)O','C(CO)CO','C(CCO)CCO','CC(CCO)CCO','CCC(CC(CC)CO)CO','CC(C)C(C(C)(C)CO)O','CCC(O)OC(CC)O','C1CC(CCC1CO)CO','C1CC(CCC1O)O','CCCCC(CC)(CO)CO','CCCC(C(CC)CO)O','C(CCCCCCO)CCCCCO','OC1CCCCCCCCC1O','CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O','C[Si](C)(O)O','C1=CC=C(C=C1)[Si](C2=CC=CC=C2)(O)O','CC(CC(C(F)(F)F)(C(F)(F)F)O)O','C(C(C(C(C(CO)(F)F)(F)F)(F)F)(F)F)O']
# cooh=['C(CCC(=O)O)CC(=O)O','C(CCCCC(=O)O)CCCC(=O)O','C1=CC(=CC=C1C(=O)O)C(=O)O','C1=CC=C2C(=C1)C(=O)OC2=O','C1=CC(=CC(=C1)C(=O)O)C(=O)O','C(CC(=O)O)C(=O)O','C(CC(=O)O)CC(=O)O','C(CCCC(=O)O)CCCC(=O)O','C(CCCCCC(=O)O)CCCCC(=O)O','C1CC(CCC1C(=O)O)C(=O)O','CCCCCCCCC(CCCCCCCCC(=O)O)C(CCCCCCCC(=O)O)CCC=CCCCCC','C=C(CC(=O)O)C(=O)O']

# reactant = Chem.MolFromSmiles("CC(C)(CO)C(=O)OCC(C)(C)C(=O)OCC(C)(C)C(=O)OCCO")
# functional_group = Chem.MolFromSmarts("[C:1](=[O:2])-[OD1]")
# matches = reactant.GetSubstructMatches(functional_group)
# print(len(matches))