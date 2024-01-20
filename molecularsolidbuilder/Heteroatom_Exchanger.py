import os, sys, subprocess, glob, random
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Chem import rdDepictor
from copy import deepcopy
from .PDBfile_Converter import PDBImageFileToMols
from .Hydrocarbons_Builder import convex_bond_atom, Find_Vertex_v2
import numpy as np
from .Utility import Embedfrom2Dto3D, Plot_2Dmol, Plot_2Dmol_tmp
from .Hydrocarbons_Builder import *
from rdkit.Chem.rdchem import HybridizationType, BondType, PeriodicTable

def Heteroatom_Func_Sub_Phenolic_OtoS(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    Os = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 0 and not atom.GetIsAromatic() \
          and atom.GetSymbol() == 'O' and len([n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and n.GetIsAromatic()]) == 1] 
    Chem.Kekulize(mol)
    if len(Os) > 0:
        random.shuffle(Os)
        #choose one
        idx_modify = Os[0]
        neighbor = [n.GetIdx() for n in atoms[idx_modify].GetNeighbors() if n.GetSymbol() == 'C' and n.GetIsAromatic()]
        if len(neighbor) == 1:
            atoms[idx_modify].SetAtomicNum(16)
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol


def num_5ringO(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'O' and (atom.IsInRingSize(5))]
    return atom_idx

def Add_OH_tofaC(mol,input_3d=False):
    mol = AllChem.RemoveHs(mol)
    AllChem.Kekulize(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    
    Hs = []
    for atom in atoms:
        sym = atom.GetSymbol()
        idx = atom.GetIdx()
        num_H = atom.GetTotalNumHs()
        nn_sym = [n.GetSymbol() for n in atom.GetNeighbors()]
        nn_o_idx = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'O']
        nn_bond_type = [mol_dup.GetBondBetweenAtoms(idx,o_idx).GetBondType() for o_idx in nn_o_idx]
        in_ring = atom.IsInRing()
        if (sym == 'C') and (num_H==1) and (len(nn_bond_type)==1) and (nn_bond_type[0] == BondType.DOUBLE):
            Hs.append(idx)

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles('O')
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        #mol_list = [AllChem.AddHs(mol,addCoords=True)]
        mol_list = [mol]
        return False, mol_list


def Single2Double(mol):
    mol = AllChem.RemoveHs(mol)
    AllChem.Kekulize(mol)
    mol = AllChem.AddHs(mol,addCoords=True)
    check = []
    bs = mol.GetBonds()
    btype = [b.GetBondType() for b in bs]
    for i, b in enumerate(bs):
        btype = b.GetBondType()
        batoms_sym = [b.GetBeginAtom().GetSymbol(),b.GetEndAtom().GetSymbol()]
        batoms_val = [b.GetBeginAtom().GetTotalValence(),b.GetEndAtom().GetTotalValence()]
        batomsH_idx = [sorted([n.GetIdx() for n in b.GetBeginAtom().GetNeighbors() if n.GetSymbol()=='H']),sorted([n.GetIdx() for n in b.GetEndAtom().GetNeighbors() if n.GetSymbol()=='H'])]
        batoms_val_woH = [a-len(b) for a, b in zip(batoms_val, batomsH_idx)]
        if (btype == BondType.SINGLE) \
            and (batoms_val_woH[0] < 4) and (batoms_val_woH[1] < 4) \
            and ((len(batomsH_idx[0]) in range(2,3)) and (len(batomsH_idx[1]) in range(2,3))) \
            and (batoms_sym[0]!='H' and batoms_sym[1]!='H'):
            check.append([b.GetIdx(), BondType.DOUBLE, batomsH_idx])
    if len(check) > 0:
        changed_mols = []
        for i, c in enumerate(check):
            mol_ = deepcopy(mol)
            h_remove = sorted([c[-1][0][-1],c[-1][1][-1]],reverse=True)
            
            EAddFunc = AllChem.EditableMol(mol_)
            for h_r in h_remove:
                EAddFunc.RemoveAtom(h_r)
            mol_ = EAddFunc.GetMol()
            bonds = mol_.GetBonds()
            bonds[c[0]].SetBondType(c[1])
            Draw.MolToFile(mol_,'test.png',highlightBonds=[c[0]])
            mol_ = AllChem.RemoveHs(mol_)
            mol2 = deepcopy(mol_)
            changed_mols.append(mol2)

        if len(changed_mols) != 0:
            return True, changed_mols
    else:
        return False, [mol]

def Double2Single(mol):
    mol = AllChem.RemoveHs(mol)
    AllChem.Kekulize(mol)
    check = []
    bs = mol.GetBonds()
    btype = [b.GetBondType() for b in bs]
    for i, b in enumerate(bs):
        btype = b.GetBondType()
        batomsH = [b.GetBeginAtom().GetTotalNumHs(),b.GetEndAtom().GetTotalNumHs()]
        if (btype == BondType.DOUBLE) and ((batomsH[0] > 0) and (batomsH[1] > 0)):
            check.append([b.GetIdx(), BondType.SINGLE])
    if len(check) == 0:
        for i, b in enumerate(bs):
            btype = b.GetBondType()
            batomsH = [b.GetBeginAtom().GetTotalNumHs(),b.GetEndAtom().GetTotalNumHs()]
            if (btype == BondType.DOUBLE):
                check.append([b.GetIdx(), BondType.SINGLE])
    changed_mols = []
    for i, c in enumerate(check):
        mol_ = deepcopy(mol)
        AllChem.Kekulize(mol_)
        bs = mol_.GetBonds()
        bs[c[0]].SetBondType(c[1])
        abi = bs[c[0]].GetBeginAtomIdx()
        aei = bs[c[0]].GetEndAtomIdx()
        atoms = mol_.GetAtoms()
        atoms[abi].SetHybridization(HybridizationType.SP3)
        atoms[abi].UpdatePropertyCache()
        atoms[aei].SetHybridization(HybridizationType.SP3)
        atoms[aei].UpdatePropertyCache()
        mol_ = AllChem.RemoveHs(mol_)
        mol2 = deepcopy(mol_)
        changed_mols.append(mol2)
    if len(changed_mols) != 0:
        return changed_mols
    else:
        return [mol]



def Heteroatom_Aromatic_Func_Add_list(mol,Func_smi,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() >= 1 and atom.IsInRing() and atom.GetIsAromatic() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    Hs = aromatic_Hs #+ aliphatic_chain_Hs + aliphatic_ring_Hs

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles(Func_smi)
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list

def Heteroatom_Aliphatic_Func_Add_list(mol,Func_smi,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    aliphatic_chain_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and (not atom.IsInRing())]
    aliphatic_ring_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and atom.IsInRing() and (not atom.GetIsAromatic())]
    Hs = aliphatic_chain_Hs + aliphatic_ring_Hs

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles(Func_smi)
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list
    
def Heteroatom_AliphaticChain_Func_Add_list(mol,Func_smi,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    aliphatic_chain_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and (not atom.IsInRing())]
    #aliphatic_ring_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and atom.IsInRing() and (not atom.GetIsAromatic())]
    Hs = aliphatic_chain_Hs

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles(Func_smi)
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list

def Heteroatom_AliphaticRing_Func_Add_list(mol,Func_smi,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    #aliphatic_chain_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and (not atom.IsInRing())]
    aliphatic_ring_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and atom.IsInRing() and (not atom.GetIsAromatic())]
    Hs = aliphatic_ring_Hs

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles(Func_smi)
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list

def Heteroatom_Aromatic_Func_Add_withSym_list(mol,symbol,Func_smi,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() and atom.GetIsAromatic() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    Hs = aromatic_Hs

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles(Func_smi)
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list

def Heteroatom_Aliphatic_Func_Add_withSym_list(mol,symbol,Func_smi,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    aliphatic_chain_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and (not atom.IsInRing()) and atom.GetSymbol() ==symbol]
    aliphatic_ring_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and atom.IsInRing() and (not atom.GetIsAromatic()) and atom.GetSymbol() ==symbol]
    Hs = aliphatic_chain_Hs + aliphatic_ring_Hs

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles(Func_smi)
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list

def Heteroatom_Func_Add_DoubleO_list(mol,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    Hs_inRing = [atom.GetIdx() for atom in atoms \
                   if atom.GetTotalNumHs() >= 2 and atom.IsInRing()]
    Hs_inChain = [atom.GetIdx() for atom in atoms \
                          if atom.GetTotalNumHs() == 3 and (not atom.IsInRing())]
    Hs = Hs_inChain #+ Hs_inRing

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles('O')
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs1_sorted = sorted(hidxs1,reverse=True)[:2]
            hidxs2_sorted = sorted(hidxs2,reverse=True)[:2]
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.DOUBLE)
            h_remove = sorted(hidxs1_sorted+hidxs2_sorted,reverse=True)
            for h_r in h_remove:
                EAddFunc.RemoveAtom(h_r)
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list

def Heteroatom_Aliphatic_Func_Add_withSym_list2(mol,symbol,Func_smi,numH=2,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    aliphatic_chain_Hs = [atom.GetIdx() for atom in atoms \
                          if (atom.GetTotalNumHs() == numH) \
                          and (not atom.IsInRing()) and atom.GetSymbol() == symbol]
    aliphatic_ring_Hs = [atom.GetIdx() for atom in atoms \
                         if (atom.GetTotalNumHs() == numH) \
                         and atom.IsInRing() and (not atom.GetIsAromatic()) \
                         and atom.GetSymbol() == symbol]
    Hs = aliphatic_chain_Hs + aliphatic_ring_Hs
    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            atoms = mol_dup.GetAtoms()            
            Func = AllChem.MolFromSmiles(Func_smi)
            Func_atoms = Func.GetAtoms()
            Func_idx = np.arange(len(Func_atoms)) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            AddFuncH = AllChem.AddHs(AddFunc,addCoords=True)
            atoms_comb = AddFuncH.GetAtoms()
            hidxs1 = [n.GetIdx() for n in atoms_comb[idx_modify].GetNeighbors() if n.GetSymbol() == 'H']
            hidxs2 = [n.GetIdx() for n in atoms_comb[int(Func_idx[0])].GetNeighbors() if n.GetSymbol() == 'H']
            EAddFunc = AllChem.EditableMol(AddFuncH)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            h_remove = sorted([hidxs1[-1],hidxs2[-1]],reverse=True)
            EAddFunc.RemoveAtom(h_remove[0])
            EAddFunc.RemoveAtom(h_remove[1])
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.AddHs(AllChem.MolFromSmiles(smi),addCoords=True), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [AllChem.AddHs(mol,addCoords=True)]
        return False, mol_list


def Heteroatom_Func_Add_OH_list(mol,input_3d=True):
    mol = AllChem.RemoveHs(mol)
    mol_dup = deepcopy(mol)
    atoms = mol_dup.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.IsInRing()]
    Hs = aromatic_Hs
    if len(Hs) > 0:
        Hs = aromatic_Hs
    else:
        Hs = aliphatic_Hs

    mol_list = []
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            #mol_dup = AllChem.RemoveHs(mol_dup)
            atoms = mol_dup.GetAtoms()
            atoms[idx_modify].SetNumExplicitHs(0)
            Func = AllChem.MolFromSmiles('O')
            Func_atoms = Func.GetAtoms()
            Func_atoms[0].SetNoImplicit(True)
            Func_atoms[0].SetNumExplicitHs(1)
            Func_idx = np.array([0]) + len(atoms)
            AddFunc = Chem.CombineMols(mol_dup,Func)
            EAddFunc = AllChem.EditableMol(AddFunc)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            mol_new = EAddFunc.GetMol()
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                smi_list.append(smi)
        if input_3d:
            mol_list = [AllChem.ConstrainedEmbed(AllChem.MolFromSmiles(smi), mol_dup) for smi in smi_list]
        else:
            mol_list = [AllChem.MolFromSmiles(smi) for smi in smi_list]
        return True, mol_list
    else:
        mol_list = [mol]
        return False, mol_list

#---------------------------------------------
def Heteroatom_Sub_Quaternary_fromCtoN_testing(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    AllChem.Kekulize(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    mol_list = []
    sites_index = [atom.GetIdx() for atom in atoms if atom.IsInRingSize(6) \
        and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'C' \
        and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) == 3]
    for i, idx in enumerate(sites_index):
        mol2 = deepcopy(mol)
        mol2 = AllChem.RemoveHs(mol2)
        atoms2 = mol2.GetAtoms()
        bonds2 = mol2.GetBonds()
        nn_idx = [n.GetIdx() for n in atoms[idx].GetNeighbors()]
        nn_bond_type = [mol.GetBondBetweenAtoms(idx,nidx).GetBondType() for nidx in nn_idx]
        [mol2.GetBondBetweenAtoms(idx,nidx).SetBondType(BondType.SINGLE) for nidx in nn_idx]
        atoms2[idx].SetAtomicNum(7)
        try:
            mol2 = AllChem.RemoveHs(mol2)
            mol_list.append(mol2)
        except:
            continue
    if len(mol_list) == 0:
        return False, [mol]
    else:
        return True, mol_list

#---------------------------------------------

def heteroatom_types(mols):
    num_phenolic_o = 0
    num_phenolic_s = 0
    num_ether_o = 0
    num_carbonyl_o = 0
    num_pyridine_n = 0
    num_pyrrolic_n = 0
    num_quatern_n = 0
    num_thiophene_s = 0

    for i, mol in enumerate(mols):
        num_phenolic_o += num_phenolic_o(mol)
        num_phenolic_s += num_phenolic_s(mol)
        num_ether_o    += num_ether_o(mol)
        num_carbonyl_o += num_carbonyl_o(mol)
        num_pyridine_n += num_pyridine_n(mol)
        num_pyrrolic_n += num_pyrrolic_n(mol)
        num_quatern_n  += num_quatern_n(mol)
        num_thiophene_s+= num_thiophene_s(mol)

    return {'num_phenolic_o':num_phenolic_o,
            'num_phenolic_s':num_phenolic_s,
            'num_ether_o':num_ether_o,
            'num_carbonyl_o':num_carbonyl_o,
            'num_pyridine_n':num_pyridine_n,
            'num_pyrrolic_n':num_pyrrolic_n,
            'num_quatern_n':num_quatern_n,
            'num_thiophene_s':num_thiophene_s}

#check and count 
def num_phenolic_o(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.GetSymbol() == 'O' and (not atom.IsInRingSize(5) or not atom.IsInRingSize(6))]
    return atom_idx

def num_phenolic_s(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.GetSymbol() == 'S' and (not atom.IsInRingSize(5) or not atom.IsInRingSize(6))]
    return atom_idx

def num_ether_o(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'O' and (atom.IsInRingSize(5) or atom.IsInRingSize(6))]
    return atom_idx

def num_carbonyl_o(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'O' and not atom.IsInRing()\
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C'])==1]
    return atom_idx

def num_pyridine_n(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'N' and atom.IsInRing()\
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C'])==2]
    return atom_idx

def num_pyrrolic_n(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.GetSymbol() == 'N' and atom.IsInRingSize(5)\
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C'])==2]
    return atom_idx

def num_quatern_n(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'N' and atom.IsInRing()\
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C'])==3]
    return atom_idx

def num_thiophene_s(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    atom_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'S' and atom.IsInRingSize(5)\
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C' and n.IsAromatic()])==2]
    return atom_idx


#############
def Convert_OH2CH3(mol):
    mol = deepcopy(mol)
    AllChem.Kekulize(mol)
    atoms = mol.GetAtoms()
    oh_atoms_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'O' and not atom.IsInRing() \
            and atom.GetTotalNumHs() == 1 and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) == 1]
    if len(oh_atoms_idx):
        for i, oh_idx in enumerate(oh_atoms_idx):
            atoms[oh_idx].SetAtomicNum(6)
            atoms[oh_idx].UpdatePropertyCache()
        mol = AllChem.RemoveHs(mol)
        #Plot_2Dmol(mol)
        return mol, True
    else:
        mol = AllChem.RemoveHs(mol)
        return mol, False

#Add aliphatic 6ring
def Heteroatom_Add_6Ring_Aliphatic(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)<= 4 and len(v) >= 2) and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']
    #vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
    #    and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v]) \
    #    and all([True if mol.GetAtoms()[t].GetTotalNumHs() <= 1 else False for t in v])]
    vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) or mol.GetAtoms()[t].IsInRingSize(5) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]

    vs1 = [v for v in vs1 if mol.GetAtoms()[v[0]].GetTotalNumHs()==1 and mol.GetAtoms()[v[-1]].GetTotalNumHs()==1]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[CH2][CH2]')
        elif len(vs) == 3:
            frg = AllChem.MolFromSmiles('[CH2][CH2][CH2]')
        else:
            frg = AllChem.MolFromSmiles('[CH2][CH2][CH2][CH2]')
        
        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        AllChem.Kekulize(fm)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

#Add aliphatic 5ring
def Heteroatom_Add_5Ring_Aliphatic(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 3 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetTotalNumHs() == 1 else False for t in v])]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[CH2]')
        elif len(vs) == 3:
            frg = AllChem.MolFromSmiles('[CH2][CH2]')
        else:
            frg = AllChem.MolFromSmiles('[CH2][CH2][CH2]')
        
        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        AllChem.Kekulize(fm)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 


#Add 5ring with sulfur for Thiophenic sulfide
def Heteroatom_Add_5Ring_S(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2 or len(v) == 3) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]
    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[S]')
        else:
            frg = AllChem.MolFromSmiles('[CH]=[CH][S]')
        
        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        AllChem.Kekulize(fm)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

#Subtitute 5ring with sulfur for Thiophenic sulfide
def Heteroatom_Sub_5Ring_fromCtoS(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 2 and atom.GetSymbol() == 'C' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(16)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Sub_5Ring_fromOtoS(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'O' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(16)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

#Add functional S
def Heteroatom_Func_Add_SH(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.GetIsAromatic() == True and atom.GetSymbol() == 'C'] 
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('S')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Remove_SH(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    OHs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and not atom.GetIsAromatic() \
          and atom.GetSymbol() == 'S' and len([n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and n.GetIsAromatic()]) == 1] 
    Chem.Kekulize(mol)
    if len(OHs) > 0:
        random.shuffle(OHs)
        #choose one
        idx_modify = OHs[0]
        neighbor = [n.GetIdx() for n in atoms[idx_modify].GetNeighbors() if n.GetSymbol() == 'C' and n.GetIsAromatic()]
        if len(neighbor) == 1:
            ERemFunc = AllChem.EditableMol(mol)
            ERemFunc.RemoveAtom(idx_modify)
            mol = ERemFunc.GetMol()
            #mol.GetAtoms()[neighbor[0]].SetNumExplicitHs(1)
            #Plot_2Dmol_tmp(mol)
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

#Add functional CS
def Heteroatom_Func_Add_CH2SH(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and atom.GetSymbol() == 'C' and len(atom.GetNeighbors()) < 3]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('[CH2][SH]')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(2)
        Func_idx = np.arange(len(Func_atoms)) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()

        [atom.SetNumRadicalElectrons(0) for atom in atoms_comb]
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

#Add 5ring with oxygen for ether
def Heteroatom_Add_5Ring_O(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[O]')
        else:
            frg = AllChem.MolFromSmiles('[CH]=[CH][O]')

        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        AllChem.Kekulize(fm)
        mol = deepcopy(fm)
        #check = deepcopy(fm)
        #AllChem.Compute2DCoords(check,nFlipsPerSample=10000)
        #Chem.Draw.MolToFile(check,'./tmp.png', size=(1000,1000))
        #subprocess.call('imgcat tmp.png',shell=True)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

#Subtitute 5ring with oxygen for ether
def Heteroatom_Sub_5Ring_fromCtoO(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 2 \
            and atom.GetSymbol() == 'C' and [n.GetSymbol() for n in atom.GetNeighbors()]==['C','C']]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(8)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Sub_6Ring_fromCtoO(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(6) and atom.GetTotalNumHs() == 2 \
            and atom.GetSymbol() == 'C' and [n.GetSymbol() for n in atom.GetNeighbors()]==['C','C']]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(8)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Sub_fal6Ring_fromCtoO(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ring = mol.GetRingInfo()
    aring = ring.AtomRings()
    aring_symbol = [[(ri, atoms[ri].GetSymbol()) for ri in ar] for ar in aring]
        
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(6) and atom.GetTotalNumHs() == 2 \
        and not atom.GetIsAromatic() and atom.GetSymbol() == 'C'\
        and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol() == 'C'])==2 \
        and len([n.GetSymbol() for n in atom.GetNeighbors() if not n.GetIsAromatic()]) == 2]

    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(8)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

#Subtitute 5ring from O to C
def Heteroatom_Sub_5Ring_fromOtoC(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'O' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetAtomicNum(6)
        atoms[idx_modify].SetNumExplicitHs(2)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

#Subtitute 6ring from O to C
def Heteroatom_Sub_6Ring_fromOtoC(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(6) and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'O' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetAtomicNum(6)
        atoms[idx_modify].SetNumExplicitHs(2)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol


def Heteroatom_CL_fromOtoS1(mol):
    mol = deepcopy(mol)
    #mol = AllChem.RemoveHs(mol)

    patt = AllChem.MolFromSmarts('cOc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    #Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[1]
            if atoms[idx_modify].GetTotalNumHs() == 2:
                atoms[idx_modify].SetNumExplicitHs(0)
                atoms[idx_modify].SetAtomicNum(16)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_CL_fromCtoS1(mol):
    mol = deepcopy(mol)
    #mol = AllChem.RemoveHs(mol)

    patt = AllChem.MolFromSmarts('cCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    #Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[1]
            if atoms[idx_modify].GetTotalNumHs() == 2:
                atoms[idx_modify].SetNumExplicitHs(0)
                atoms[idx_modify].SetAtomicNum(16)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_CL_fromStoO2(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCSCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            if atoms[idx_modify].GetTotalNumHs() == 2:
                atoms[idx_modify].SetNumExplicitHs(0)
                atoms[idx_modify].SetAtomicNum(8)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_CL_fromOtoS2(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCOCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            if atoms[idx_modify].GetTotalNumHs() == 2:
                atoms[idx_modify].SetNumExplicitHs(0)
                atoms[idx_modify].SetAtomicNum(16)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_CL_fromCtoS2(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCCCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            if atoms[idx_modify].GetTotalNumHs() == 2:
                atoms[idx_modify].SetNumExplicitHs(0)
                atoms[idx_modify].SetAtomicNum(16)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol


#Subtitute 5ring from O to C
def Heteroatom_Sub_5Ring_fromStoC(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'S' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(2)
        atoms[idx_modify].SetAtomicNum(6)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_CL_fromStoC2(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCSCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            if atoms[idx_modify].GetTotalNumHs() == 0:
                atoms[idx_modify].SetNumExplicitHs(2)
                atoms[idx_modify].SetAtomicNum(6)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_CL_fromStoC1(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cSc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    #Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[1]
            if atoms[idx_modify].GetTotalNumHs() == 0:
                atoms[idx_modify].SetNumExplicitHs(2)
                atoms[idx_modify].SetAtomicNum(6)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol


def Heteroatom_CL_fromOtoC2(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCOCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            if atoms[idx_modify].GetTotalNumHs() == 0:
                atoms[idx_modify].SetNumExplicitHs(2)
                atoms[idx_modify].SetAtomicNum(6)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_CL_fromOtoC1(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cOc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    #Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[1]
            if atoms[idx_modify].GetTotalNumHs() == 0:
                atoms[idx_modify].SetNumExplicitHs(2)
                atoms[idx_modify].SetAtomicNum(6)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol


#Add 6ring with nitrogen for Pyridinic
def Heteroatom_Add_6Ring_N(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']
    vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[CH]=[N]')
        else:
            frg = AllChem.MolFromSmiles('[CH]=[N][CH]=[CH]')

        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        AllChem.Kekulize(fm)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 




#Add 5ring with nitrogen for Pyrrolic
def Heteroatom_Add_5Ring_N(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]


    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[NH]')
        else:
            frg = AllChem.MolFromSmiles('[CH]=[CH][NH]')

        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        AllChem.Kekulize(fm)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

def Heteroatom_Add_5Ring_C(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[CH2]')
        else:
            smi_choice = random.choice(['[CH]=[CH][CH2]','[CH2][CH]=[CH]'])
            frg = AllChem.MolFromSmiles(smi_choice)

        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        AllChem.Kekulize(fm)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

def Heteroatom_Add_6Ring_C(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if (mol.GetAtoms()[v[0]].GetTotalNumHs() == 1 and mol.GetAtoms()[v[-1]].GetTotalNumHs() == 1)\
        and all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        if len(vs) == 4:
            frg = AllChem.MolFromSmiles('[CH]=[CH]')
        else:
            frg = AllChem.MolFromSmiles('[CH]=[CH][CH]=[CH]')

        mcomb = Chem.CombineMols(mol,frg)
        mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
        mcomb_atoms = mcomb.GetAtoms()

        mcomb = AllChem.AddHs(mcomb)
        main_idx = mcomb_idx[:-len(frg.GetAtoms())]
        frg_idx = mcomb_idx[-len(frg.GetAtoms()):]
        edcombo = Chem.EditableMol(mcomb)
        edcombo.AddBond(frg_idx[0],vs[0],order=Chem.rdchem.BondType.SINGLE)
        edcombo.AddBond(frg_idx[-1],vs[-1],order=Chem.rdchem.BondType.SINGLE)
        ht = []
        for vi in range(len(vs)):
            hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[vs[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
            if len(hs) > 0:
                ht += hs
        ht = sorted(ht,reverse=True)
        [edcombo.RemoveAtom(t) for t in ht]
        fm = edcombo.GetMol()
        fm = AllChem.RemoveHs(fm)
        atoms2 = fm.GetAtoms()
        bonds2 = fm.GetBonds()
        [atom.SetNumRadicalElectrons(0) for atom in atoms2]
        try:
            AllChem.Kekulize(fm)
        except:
            Plot_2Dmol(mol)
            Plot_2Dmol_tmp(fm)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 


#Subtitute 6ring with nitrogen for Pyridinic
def Heteroatom_Sub_6Ring_fromCtoN(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(6) and atom.GetTotalNumHs() == 1 and atom.GetSymbol() == 'C' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(7)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol




def Heteroatom_Sub_6Ring_fromNtoC(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(6) and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'N' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(1)
        atoms[idx_modify].SetAtomicNum(6)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol



#Subtitute 5ring with nitrogen for Pyrrolic
def Heteroatom_Sub_5Ring_fromCtoN(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 2 and atom.GetSymbol() == 'C' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(7)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Sub_5Ring_fromOtoN(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'O' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(1)
        atoms[idx_modify].SetAtomicNum(7)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol    

def Heteroatom_Sub_5Ring_fromNtoC(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 1 and atom.GetSymbol() == 'N' \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) >=2 ]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetAtomicNum(6)
        atoms[idx_modify].SetNumExplicitHs(2)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Sub_Quaternary_fromCtoN(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms if atom.IsInRingSize(6) \
        and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'C' \
        and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) == 3 \
        and atom.GetTotalValence() == 3]
    print(sites_index)
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetAtomicNum(7)
        atoms[idx_modify].SetNumExplicitHs(0)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Sub_Quaternary_fromNtoC(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms if atom.IsInRingSize(6) \
        and atom.GetTotalNumHs() == 0 and atom.GetSymbol() == 'N' \
        and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) == 3 \
        and atom.GetTotalValence() == 3]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetAtomicNum(6)
        atoms[idx_modify].SetNumExplicitHs(0)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol


#Aliphaptic (dibenzul sulfide) (sulfur crosslinking?), oxygen crosslinking
def Heteroatom_CL_fromCtoO1(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[1]
            if atoms[idx_modify].GetTotalNumHs() == 2 and not atoms[idx_modify].IsInRing():
                atoms[idx_modify].SetAtomicNum(8)
                atoms[idx_modify].SetNumExplicitHs(0)
                mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol
    
def Heteroatom_CL_fromCtoO2(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCCCc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    #Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            if atoms[idx_modify].GetTotalNumHs() == 2 and not atoms[idx_modify].IsInRing():
                atoms[idx_modify].SetNoImplicit(True)
                atoms[idx_modify].SetNumExplicitHs(0)
                atoms[idx_modify].SetAtomicNum(8)
                #mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                #mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        #mol = AllChem.RemoveHs(mol)
        return False, mol

#Functional group --> substitute aromatic hydrogens


#propyl -CCH2CH3
def Heteroatom_Func_Propyl(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing() and atom.GetSymbol() == 'C']
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('CCC')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(2)
        Func_idx = np.array([0,1,2]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

#2propyl -CC(CH3)CH3
def Heteroatom_Func_2Propyl(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing()]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('CC(C)C')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(2)
        Func_idx = np.array([0,1,2,3]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

###


def Crosslink_convert_cOc_cCOCc(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cOc')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    oneChoice = hit_ats[0][1:-1]
    repl = AllChem.MolFromSmarts('COC')


    return oneChoice


def Heteroatom_Func_Add_CH2OH(mol):
    mol = deepcopy(mol)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing() if atom.GetSymbol() == 'C']

    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('CO')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(2)
        Func_idx = np.array([0,1]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol


#Ethyl -CH2CH3
def Heteroatom_Func_Add_CH2CH3(mol):
    mol = deepcopy(mol)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing() if atom.GetSymbol() == 'C']

    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('CC')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(2)
        Func_idx = np.array([0,1]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Add_OH_al1(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.IsInRing()]
    Hs = aliphatic_Hs
    if len(Hs) > 0:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol)
    if len(Hs) > 0:
        random.shuffle(Hs)
        #choose one
        idx_modify = Hs[0]
        atoms[idx_modify].SetNumExplicitHs(atoms[idx_modify].GetTotalNumHs()-1)
        Func = AllChem.MolFromSmiles('O')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Add_OH_al2(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if (atom.GetTotalNumHs() <= 2 and atom.GetTotalNumHs() > 1) and not atom.GetIsAromatic() and atom.IsInRing()]
    Hs = aliphatic_Hs
    if len(Hs) > 0:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol)
    if len(Hs) > 0:
        random.shuffle(Hs)
        #choose one
        idx_modify = Hs[0]
        atoms[idx_modify].SetNumExplicitHs(atoms[idx_modify].GetTotalNumHs()-1)
        Func = AllChem.MolFromSmiles('O')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

### Functional group for aliphaic carbon
def Heteroatom_Func_Add_CH3_al1(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.GetIsAromatic() and not atom.IsInRing()]
    Hs = aliphatic_Hs
    if len(Hs) > 0:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol)
    if len(Hs) > 0:
        random.shuffle(Hs)
        #choose one
        idx_modify = Hs[0]
        atoms[idx_modify].SetNumExplicitHs(atoms[idx_modify].GetTotalNumHs()-1)
        Func = AllChem.MolFromSmiles('C')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(3)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol


def Heteroatom_Func_Add_CH3_al2(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    #aromatic_Hs = [atom.GetIdx() for atom in atoms \
    #    if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
    #    and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if (atom.GetTotalNumHs() == 2) and not atom.GetIsAromatic() and atom.IsInRing()]
    Hs = aliphatic_Hs
    if len(Hs) > 0:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol)
    if len(Hs) > 0:
        random.shuffle(Hs)
        #choose one
        idx_modify = Hs[0]
        atoms[idx_modify].SetNumExplicitHs(atoms[idx_modify].GetTotalNumHs()-1)
        Func = AllChem.MolFromSmiles('C')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(3)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol


def Heteroatom_Func_Add_CH3_al3(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    #aromatic_Hs = [atom.GetIdx() for atom in atoms \
    #    if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
    #    and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if (atom.GetTotalNumHs() <= 2 and atom.GetTotalNumHs() >= 1) and not atom.GetIsAromatic() and atom.IsInRing()]
    Hs = aliphatic_Hs
    if len(Hs) > 0:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol)
    if len(Hs) > 0:
        random.shuffle(Hs)
        #choose one
        idx_modify = Hs[0]
        atoms[idx_modify].SetNumExplicitHs(atoms[idx_modify].GetTotalNumHs()-1)
        Func = AllChem.MolFromSmiles('C')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(3)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol


### Functional group for aromatic carbon
def Heteroatom_Func_Add_CH3(mol,input_3d=True):
    mol_new = deepcopy(mol)
    mol_new = AllChem.RemoveHs(mol_new)
    atoms = mol_new.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.IsInRing()]
    Hs = aromatic_Hs
    if len(Hs) > 0:
        Hs = aromatic_Hs
    else:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol_new)
    if len(Hs) > 0:
        random.shuffle(Hs)
        #choose one
        idx_modify = Hs[0]
        atoms[idx_modify].SetNumExplicitHs(atoms[idx_modify].GetTotalNumHs()-1)
        Func = AllChem.MolFromSmiles('C')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(3)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol_new,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol_new = EAddFunc.GetMol()
        atoms_comb = mol_new.GetAtoms()
        if input_3d:
            AllChem.ConstrainedEmbed(mol_new, mol)
        else:
            mol_new = Embedfrom2Dto3D(mol_new)

        mol_new = AllChem.RemoveHs(mol_new)
        return True, mol_new
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Remove_CH3(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    CH3s = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.GetIsAromatic() \
          and atom.GetSymbol() == 'C' and len([n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and n.GetIsAromatic()]) == 1] 
    Chem.Kekulize(mol)
    if len(CH3s) > 0:
        random.shuffle(CH3s)
        #choose one
        idx_modify = CH3s[0]
        neighbor = [n.GetIdx() for n in atoms[idx_modify].GetNeighbors() if n.GetSymbol() == 'C' and n.GetIsAromatic()]
        if len(neighbor) == 1:
            ERemFunc = AllChem.EditableMol(mol)
            ERemFunc.RemoveAtom(idx_modify)
            mol = ERemFunc.GetMol()
            #mol.GetAtoms()[neighbor[0]].SetNumExplicitHs(1)
            #Plot_2Dmol_tmp(mol)
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Add_OH(mol,input_3d=True):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    mol = AllChem.AddHs(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.GetIsAromatic() == True and atom.GetSymbol() == 'C'] 
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        mol_new = deepcopy(mol)
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('O')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol_new,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol_new = EAddFunc.GetMol()
        atoms_comb = mol_new.GetAtoms()
        if input_3d:
            AllChem.ConstrainedEmbed(mol_new, mol)
        else:
            mol_new = Embedfrom2Dto3D(mol_new)

        mol_new = AllChem.RemoveHs(mol_new)
        return True, mol_new
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Remove_OH(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    OHs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and not atom.GetIsAromatic() \
          and atom.GetSymbol() == 'O' and len([n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and n.GetIsAromatic()]) == 1] 
    Chem.Kekulize(mol)
    if len(OHs) > 0:
        random.shuffle(OHs)
        #choose one
        idx_modify = OHs[0]
        neighbor = [n.GetIdx() for n in atoms[idx_modify].GetNeighbors() if n.GetSymbol() == 'C' and n.GetIsAromatic()]
        if len(neighbor) == 1:
            ERemFunc = AllChem.EditableMol(mol)
            ERemFunc.RemoveAtom(idx_modify)
            mol = ERemFunc.GetMol()
            #mol.GetAtoms()[neighbor[0]].SetNumExplicitHs(1)
            #Plot_2Dmol_tmp(mol)
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

##Functional group for aliphatic carbon
def Heteroatom_Func_Carbonyl_SubCH2toO(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    aliphatic_C = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 2 and atom.IsInRing() \
        and atom.GetSymbol() == 'C' and [n.GetSymbol() for n in atom.GetNeighbors()] == ['C','C'] ]
    Chem.Kekulize(mol)
    if len(aliphatic_C) > 0:
        random.shuffle(aliphatic_C)
        idx_modify = aliphatic_C[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('[O]')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(0)
        Func_idx = np.arange(len(Func_atoms)) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.DOUBLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()

        [atom.SetNumRadicalElectrons(0) for atom in atoms_comb]
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Carbonyl_Aldehyde(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()

    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() <= 2 \
        and (atom.IsInRingSize(6) or atom.IsInRingSize(5)) \
        and atom.GetSymbol() == 'C' and len([n.GetSymbol() for n in atom.GetNeighbors()]) <= 2]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('C(=O)')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
        Func_idx = np.array([0,1]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Carbonyl_Ketone(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()

    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() <= 2 \
        and (atom.IsInRingSize(6) or atom.IsInRingSize(5)) \
        and atom.GetSymbol() == 'C' and len([n.GetSymbol() for n in atom.GetNeighbors()]) <= 2]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('C(=O)C')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(0)
        Func_idx = np.array([0,1,2]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Carbonyl_Ketone2(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()

    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() <= 2 \
        and (atom.IsInRingSize(6) or atom.IsInRingSize(5)) \
        and atom.GetSymbol() == 'C' and len([n.GetSymbol() for n in atom.GetNeighbors()]) <= 2]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('C(=O)C')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(0)
        Func_idx = np.array([0,1,2]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Carbonyl_Carboxyl(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()

    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() <= 2 \
        and (atom.IsInRingSize(6) or atom.IsInRingSize(5)) \
        and atom.GetSymbol() == 'C' and len([n.GetSymbol() for n in atom.GetNeighbors()]) <= 2]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('C(=O)O')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(0)
        Func_idx = np.array([0,1,2]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Add_6Ring_C_v2(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if (mol.GetAtoms()[v[0]].GetTotalNumHs() == 1 and mol.GetAtoms()[v[-1]].GetTotalNumHs() == 1)\
        and all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        mols_cycle = Propagate_v2(mol,vs1,True,False,0)
        fm = random.choice(mols_cycle)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

def Heteroatom_Add_5Ring_C_v2(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
            mol.GetAtoms()[v[0]].GetSymbol() == 'C' and mol.GetAtoms()[v[-1]].GetSymbol() == 'C']

    vs1 = [v for v in vs1 if (mol.GetAtoms()[v[0]].GetTotalNumHs() == 1 and mol.GetAtoms()[v[-1]].GetTotalNumHs() == 1)\
        and all([True if mol.GetAtoms()[t].IsInRingSize(6) else False for t in v]) \
        and all([True if mol.GetAtoms()[t].GetSymbol() =='C' else False for t in v])]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        mols_cycle = Propagate_v2(mol,vs1,False,True,0)
        fm = random.choice(mols_cycle)
        mol = deepcopy(fm)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

def Heteroatom_Func_SubOHtoSH(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    OHs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and not atom.GetIsAromatic() \
          and atom.GetSymbol() == 'O' and len([n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and n.GetIsAromatic()]) == 1] 
    Chem.Kekulize(mol)
    if len(OHs) > 0:
        random.shuffle(OHs)
        #choose one
        idx_modify = OHs[0]
        neighbor = [n.GetIdx() for n in atoms[idx_modify].GetNeighbors() if n.GetSymbol() == 'C' and n.GetIsAromatic()]
        if len(neighbor) == 1:
            atoms[idx_modify].SetAtomicNum(16)
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Func_Add_CH3_list(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.IsInRing()]
    Hs = aromatic_Hs
    if len(Hs) > 0:
        Hs = aromatic_Hs
    else:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol)

    mol_list = []
    if len(Hs) > 0:
        #random.shuffle(Hs)
        #choose one
        #idx_modify = Hs[0]
        for idx_modify in Hs:
            mol_new = deepcopy(mol)
            mol_new = AllChem.RemoveHs(mol_new)
            atoms = mol.GetAtoms()
            atoms[idx_modify].SetNumExplicitHs(atoms[idx_modify].GetTotalNumHs()-1)
            Func = AllChem.MolFromSmiles('C')
            Func_atoms = Func.GetAtoms()
            Func_atoms[0].SetNoImplicit(True)
            Func_atoms[0].SetNumExplicitHs(3)
            Func_idx = np.array([0]) + len(atoms)
            AddFunc = Chem.CombineMols(mol_new,Func)
            EAddFunc = AllChem.EditableMol(AddFunc)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            mol_new = EAddFunc.GetMol()
            atoms_comb = mol_new.GetAtoms()
            mol_new = AllChem.RemoveHs(mol_new)
            mol_list.append(mol_new)
        return True, mol_list
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol_list
    
def Heteroatom_Func_Add_CH2CH3_list(mol,input_3d=True):
    mol = deepcopy(mol)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms \
        if atom.GetTotalNumHs() == 1 and atom.IsInRing() \
        and ('N' not in [n.GetSymbol() for n in atom.GetNeighbors()]) ]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.IsInRing()]
    Hs = aromatic_Hs
    if len(Hs) > 0:
        Hs = aromatic_Hs
    else:
        Hs = aliphatic_Hs
    Chem.Kekulize(mol)

    mol_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            mol_new = deepcopy(mol)
            mol_new = AllChem.RemoveHs(mol_new)
            atoms = mol.GetAtoms()
            atoms[idx_modify].SetNumExplicitHs(0)
            Func = AllChem.MolFromSmiles('CC')
            Func_atoms = Func.GetAtoms()
            Func_atoms[0].SetNoImplicit(True)
            Func_atoms[0].SetNumExplicitHs(2)
            Func_idx = np.array([0,1]) + len(atoms)
            AddFunc = Chem.CombineMols(mol,Func)
            EAddFunc = AllChem.EditableMol(AddFunc)
            EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
            mol_new = EAddFunc.GetMol()
            atoms_comb = mol_new.GetAtoms()
            if input_3d:
                AllChem.ConstrainedEmbed(mol_new, mol)
            else:
                mol_new = Embedfrom2Dto3D(mol_new)

            mol_new = AllChem.RemoveHs(mol_new)
            mol_list.append(mol_new)
        return True, mol_list
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol_list


    
"""
def Heteroatom_Func_Add_OH(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.GetIsAromatic() == True and atom.GetSymbol() == 'C'] 
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('O')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.SINGLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

"""