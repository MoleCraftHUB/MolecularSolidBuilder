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


def Convert_Aromatic2Aliphatic(mol):
    mol = deepcopy(mol)
    mol = AllChem.RemoveHs(mol)
    AllChem.Kekulize(mol)
    bonds = mol.GetBonds()
    ar_bonds_idx = [bond.GetIdx() for bond in bonds \
        if bond.GetIsAromatic() and bond.IsInRing() and not bond.IsInRingSize(5) and \
            (bond.GetBeginAtom().GetTotalNumHs() == 1 and bond.GetEndAtom().GetTotalNumHs() == 1) and \
            (bond.GetBeginAtom().GetSymbol() == 'C' and bond.GetEndAtom().GetSymbol() == 'C')    ]
    #print(len(ar_bonds_idx), len(bonds))

    if len(ar_bonds_idx) > 0:
        abond = random.choice(ar_bonds_idx)

        b = bonds[abond]
        a1 = bonds[abond].GetBeginAtom()
        a2 = bonds[abond].GetEndAtom()

        b.SetBondType(Chem.rdchem.BondType.SINGLE)
        a1.SetNumExplicitHs(1)
        a2.SetNumExplicitHs(1)
        a1.UpdatePropertyCache()
        a2.UpdatePropertyCache()
        mol = AllChem.RemoveHs(mol)
        return True, mol
    else:
        mol = AllChem.RemoveHs(mol)
        return False, mol

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

def Heteroatom_Func_Add_OH_list(mol,input_3d=True):
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
    smi_list = []
    if len(Hs) > 0:
        for idx_modify in Hs:
            mol_new = deepcopy(mol)
            mol_new = AllChem.RemoveHs(mol_new)
            atoms = mol.GetAtoms()
            atoms[idx_modify].SetNumExplicitHs(0)
            Func = AllChem.MolFromSmiles('O')
            Func_atoms = Func.GetAtoms()
            Func_atoms[0].SetNoImplicit(True)
            Func_atoms[0].SetNumExplicitHs(1)
            Func_idx = np.array([0]) + len(atoms)
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
            smi = AllChem.MolToSmiles(mol_new)
            if smi not in smi_list:
                mol_list.append(AllChem.RemoveHs(mol_new))
                smi_list.append(smi)
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