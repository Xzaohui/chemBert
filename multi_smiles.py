oh=['CCC(C)CC','C1CCC(CC1)','CC1CCC(CC1)C','CC','C(C)C']
cooh=['CCCC','CCCCCCCC','c1cccc(c1)','c1ccccc1','c1ccc(cc1)']

for i in range(len(oh)):
    oh[i]='O'+oh[i]+'O'

for i in range(len(cooh)):
    cooh[i]='O=C(O)'+cooh[i]+'C(=O)O'

toh=oh[:]
tcooh=cooh[:]


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]-[C!H0:4]>>[C:1](=[O:2])-[O:3]-[C:4]')


reactants = (AllChem.MolFromSmiles('CCCC'), AllChem.MolFromSmiles('OCCO'))
products = rxn.RunReactants(reactants)
for p in products:
    print(AllChem.MolToSmiles(p[0]))

res=set()
for i in cooh:
    for j in oh:
        reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
        products = rxn.RunReactants(reactants)
        for p in products:
            smi = AllChem.MolToSmiles(p[0])
            res.add(smi)
            break

oh+=list(res)
cooh+=list(res)

# print(list(res))

res=set()
for i in cooh:
    for j in oh:
        reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
        products = rxn.RunReactants(reactants)
        for p in products:
            smi = AllChem.MolToSmiles(p[0])
            res.add(smi)
            break



oh+=list(res)
oh=list(set(oh))
cooh+=list(res)
cooh=list(set(cooh))

# print(list(res))

t=list(res)
res=set()
for i in t:
    for j in toh:
        reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
        products = rxn.RunReactants(reactants)
        for p in products:
            smi = AllChem.MolToSmiles(p[0])
            res.add(smi)
            break

t=list(res)
res=set()
for i in t:
    for j in toh:
        reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
        products = rxn.RunReactants(reactants)
        for p in products:
            smi = AllChem.MolToSmiles(p[0])
            res.add(smi)
            break

print(list(res))


res=set()
for i in cooh:
    for j in oh:
        reactants = (AllChem.MolFromSmiles(i), AllChem.MolFromSmiles(j))
        products = rxn.RunReactants(reactants)
        for p in products:
            smi = AllChem.MolToSmiles(p[0])
            res.add(smi)

print(list(res))

mol = Chem.MolFromSmiles('CC(CO)OC(=O)CCCCCCCCC(=O)OCCOC(=O)CCCCC(=O)OC1CCC(O)CC1')
Draw.MolToImage(mol, size=(300,300), kekulize=True)
Draw.ShowMol(mol, size=(300,300), kekulize=False)