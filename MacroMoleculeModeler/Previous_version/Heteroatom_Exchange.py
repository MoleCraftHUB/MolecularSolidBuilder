import os, sys, subprocess, glob
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from PDBfile_importer import PDBImageFileToMols
from Build_HC_revise import convex_bond_atom, Find_Vertex_v2
from rdkit.Chem import rdDepictor
from copy import deepcopy
import random
import numpy as np

#Thiophenic sulfide
def Heteroatom_Add_5Ring_S(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 ]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        print(vs)
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
        #check = deepcopy(fm)
        #AllChem.Compute2DCoords(check,nFlipsPerSample=10000)
        #Chem.Draw.MolToFile(check,'./tmp.png', size=(1000,1000))
        #subprocess.call('imgcat tmp.png',shell=True)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol 

def Heteroatom_Func_S(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing()]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('[SH]')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
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

def Heteroatom_Func_CS(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing()]
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


def Heteroatom_Add_5Ring_O(mol):
    mol = deepcopy(mol)

    vs1 = [v for v in Find_Vertex_v2(mol) if (len(v)== 4 or len(v) == 2) and \
			mol.GetAtoms()[v[0]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 and \
			mol.GetAtoms()[v[-1]].GetHybridization()==Chem.rdchem.HybridizationType.SP2 ]

    if len(vs1)> 0:
        vs1 = sorted(vs1,key=lambda x:len(x),reverse=True)
        vs = vs1[0]
        print(vs)
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


def Heteroatom_Func_O(mol):
    mol = deepcopy(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing()]
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('[OH]')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(1)
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

#Ring substitution
#Thiol


def Heteroatom_Ring_DibenzoTh_S(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('c1ccc2c(c1)Cc1ccccc1-2')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        hit_at = hit_ats[0]
        hit_atoms = list(hit_at)
        hit_bonds = []

        for bond in patt.GetBonds():
            aid1 = hit_atoms[bond.GetBeginAtomIdx()]
            aid2 = hit_atoms[bond.GetEndAtomIdx()]
            hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

        idx_modify = hit_atoms[6]
        #print( atoms[idx_modify].GetNumExplicitHs(), atoms[idx_modify].GetNumImplicitHs() )
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(16)
        mol = AllChem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return True, mol
    else:
        Chem.SanitizeMol(mol)
        return False, mol

def Heteroatom_Ring_BenzoTh_S(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('C1=Cc2ccccc2C1')
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

            idx_modify = hit_atoms[8]
            #print( atoms[idx_modify].GetNumExplicitHs(), atoms[idx_modify].GetNumImplicitHs() )
            atoms[idx_modify].SetNumExplicitHs(0)
            atoms[idx_modify].SetAtomicNum(16)
            mol = AllChem.RemoveHs(mol)

        return True, mol
    else:
        return False, mol





def Heteroatom_5Ring_S(mol):
    mol = deepcopy(mol)

    #patt = AllChem.MolFromSmarts('c1cCcc1')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    sites_index = [atom.GetIdx() for atom in atoms \
        if atom.IsInRingSize(5) and atom.GetTotalNumHs() == 2 and all([n.GetHybridization() == Chem.rdchem.HybridizationType.SP2 for n in atom.GetNeighbors()])]
    random.shuffle(sites_index)
    Chem.Kekulize(mol)
    print(sites_index)
    if len(sites_index) > 0:
        idx_modify = sites_index[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        atoms[idx_modify].SetAtomicNum(16)
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Ring_Thiophic_S(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('c1cCcc1')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    hit_ats = list(mol.GetSubstructMatches(patt))
    print(hit_ats)
    random.shuffle(hit_ats)
    Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            #print( atoms[idx_modify].GetNumExplicitHs(), atoms[idx_modify].GetNumImplicitHs() )
            atoms[idx_modify].SetNumExplicitHs(0)
            atoms[idx_modify].SetAtomicNum(16)
            mol = AllChem.RemoveHs(mol)
            return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Ring_Pyrrolic_N(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('c1cCcc1')
    #patt2 = AllChem.MolFromSmarts('C1=Cc2ccccc2C1')
    patt2 = AllChem.MolFromSmarts('c1CC=Cc1')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    hit_ats2 = list(mol.GetSubstructMatches(patt2))
    random.shuffle(hit_ats2)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    Chem.Kekulize(mol)
    #print(hit_ats, hit_ats2)
    if len(hit_ats) > 0:
        for hit_at in hit_ats:
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[2]
            #print( atoms[idx_modify].GetNumExplicitHs(), atoms[idx_modify].GetNumImplicitHs() )
            atoms[idx_modify].SetNumExplicitHs(1)
            atoms[idx_modify].SetAtomicNum(7)
            #Chem.SanitizeMol(mol)
            mol = AllChem.RemoveHs(mol)
            return True, mol
    elif len(hit_ats2) > 0:
        for hit_at in hit_ats2:
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[1]
            atoms[idx_modify].SetNumExplicitHs(0)
            atoms[idx_modify].SetAtomicNum(7)
            #print( atoms[idx_modify].GetNumExplicitHs(), atoms[idx_modify].GetNumImplicitHs() )

            #Chem.SanitizeMol(mol)
            mol = AllChem.RemoveHs(mol)
            return True, mol
    else:
        #Chem.SanitizeMol(mol)
        mol = AllChem.RemoveHs(mol)
        return False, mol

def Heteroatom_Ring_Pyridinic_N(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('c1ccccc1')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    #Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for hit_at in hit_ats:
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            for idx_modify in hit_atoms:
                if atoms[idx_modify].GetTotalNumHs() == 1:
                    atoms[idx_modify].SetNumExplicitHs(0)
                    atoms[idx_modify].SetAtomicNum(7)
                    #Chem.SanitizeMol(mol)
                    mol = AllChem.RemoveHs(mol)
                    return True, mol
                else:
                    continue
    else:
        mol = AllChem.RemoveHs(mol)
        #Chem.SanitizeMol(mol)
        return False, mol

def Heteroatom_Ring_Quaternary_N(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('c1ccc2ccccc2c1')
    #patt = AllChem.MolFromSmarts('C1C=CC2=CC=CC3=C2C1=CC=C3')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)

    Chem.Kekulize(mol)
    #AllChem.Compute2DCoords(mol,nFlipsPerSample=10)
    #Chem.rdCoordGen.AddCoords(mol)
    #Chem.Draw.MolToFile(mol,'./tmp0.png', size=(800,800), kekulize=True)
    #print([bond.GetBondType() for bond in bonds])
    if len(hit_ats) > 0:
        for i, hit_at in enumerate(hit_ats):
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            idx_modify = hit_atoms[3]
            bs = atoms[idx_modify].GetBonds()
            atoms[idx_modify].SetIsAromatic(False)
            #atoms[idx_modify].SetHybridization(Chem.rdchem.HybridizationType.SP3)
            [b1.SetBondType(Chem.rdchem.BondType.SINGLE) for b1 in bs]
            [b1.SetIsAromatic(False) for b1 in bs]
            #print(atoms[idx_modify].GetIsAromatic(), [b1.GetBondType() for b1 in bs])
            #atoms[idx_modify].SetNumExplicitHs(0)
            atoms[idx_modify].SetAtomicNum(7)
            atoms[idx_modify].UpdatePropertyCache(True)
            mol = AllChem.RemoveHs(mol)
            return True, mol
    else:
        return False, mol

def Heteroatom_Ring_Ether_O(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('c1ccccc1')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    hit_ats = list(mol.GetSubstructMatches(patt))
    random.shuffle(hit_ats)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    Chem.Kekulize(mol)
    if len(hit_ats) > 0:
        for hit_at in hit_ats:
            hit_atoms = list(hit_at)
            hit_bonds = []

            for bond in patt.GetBonds():
                aid1 = hit_atoms[bond.GetBeginAtomIdx()]
                aid2 = hit_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            for idx_modify in hit_atoms:
                if atoms[idx_modify].GetTotalNumHs() == 1:
                    atoms[idx_modify].SetIsAromatic(False)
                    [b.SetBondType(Chem.rdchem.BondType.SINGLE) for b in atoms[idx_modify].GetBonds()] 
                    [b.SetIsAromatic(False) for b in atoms[idx_modify].GetBonds()]
                    atoms[idx_modify].SetNumExplicitHs(0)
                    atoms[idx_modify].SetAtomicNum(8)
                    #Chem.SanitizeMol(mol)
                    mol = AllChem.RemoveHs(mol)
                    return True, mol
                else:
                    continue
    else:
        mol = AllChem.RemoveHs(mol)
        #Chem.SanitizeMol(mol)
        return False, mol

def Heteroatom_Ring_Ester_O(mol):
    mol = deepcopy(mol)

    patt1 = AllChem.MolFromSmarts('c1cCcc1')
    patt2 = AllChem.MolFromSmarts('C1=CCcc1')
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    hit_ats = []
    hit_ats1 = list(mol.GetSubstructMatches(patt1))
    if len(hit_ats1) != 0:
        hit_ats = hit_ats1
        patt = patt1
    else:
        hit_ats2 = list(mol.GetSubstructMatches(patt2))
        hit_ats = hit_ats2
        patt = patt2
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

            idx_modify = hit_atoms[2]
            #print( atoms[idx_modify].GetNumExplicitHs(), atoms[idx_modify].GetNumImplicitHs() )
            atoms[idx_modify].SetNumExplicitHs(0)
            atoms[idx_modify].SetAtomicNum(8)
            mol = AllChem.RemoveHs(mol)
            return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

#Aliphaptic (dibenzul sulfide) (sulfur crosslinking?), oxygen crosslinking
def Heteroatom_CL_O1(mol):
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
            if atoms[idx_modify].GetTotalNumHs() == 2:
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

def Heteroatom_CL_O2(mol):
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
            if atoms[idx_modify].GetTotalNumHs() == 2:
                atoms[idx_modify].SetNumExplicitHs(0)
                atoms[idx_modify].SetAtomicNum(8)
                mol = AllChem.RemoveHs(mol)
                return True, mol
            else:
                mol = AllChem.RemoveHs(mol)
                return False, mol

    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol
    
def Heteroatom_CL_S2(mol):
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

def Heteroatom_CL_S1(mol):
    mol = deepcopy(mol)

    patt = AllChem.MolFromSmarts('cCc')
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

#Functional group --> substitute aromatic hydrogens

#Phenolic Oxygen (-OH), Carbonyl (=O), Carboxyl (-C(=O)OH)
"""
def Heteroatom_Func_OH(mol):
    mol = deepcopy(mol)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
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

def Heteroatom_Func_O(mol):
    mol = deepcopy(mol)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 2 and atom.IsInRingSize(6) and atom.GetSymbol() == 'C']
    Chem.Kekulize(mol)
    if len(aromatic_Hs) > 0:
        random.shuffle(aromatic_Hs)
        #choose one
        idx_modify = aromatic_Hs[0]
        atoms[idx_modify].SetNumExplicitHs(0)
        Func = AllChem.MolFromSmiles('O')
        Func_atoms = Func.GetAtoms()
        Func_atoms[0].SetNoImplicit(True)
        Func_atoms[0].SetNumExplicitHs(0)
        Func_idx = np.array([0]) + len(atoms)
        AddFunc = Chem.CombineMols(mol,Func)
        EAddFunc = AllChem.EditableMol(AddFunc)
        EAddFunc.AddBond(idx_modify,int(Func_idx[0]),order=Chem.rdchem.BondType.DOUBLE)
        mol = EAddFunc.GetMol()
        atoms_comb = mol.GetAtoms()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol
"""



def Heteroatom_Func_Carboxyl(mol):
    mol = deepcopy(mol)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    atoms = mol.GetAtoms()

    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRingSize(6) and atom.GetSymbol() == 'C']
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

#Methyl -CH3
def Heteroatom_Func_Methyl(mol):
    mol = deepcopy(mol)
    convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
    atoms = mol.GetAtoms()
    aromatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and atom.IsInRing()]
    aliphatic_Hs = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 3 and not atom.IsInRing()]
    Hs = aromatic_Hs
    if len(Hs) > 0:
        Hs = aromatic_Hs
    else:
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

#Ethyl -CH2CH3
def Heteroatom_Func_Ethyl(mol):
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

