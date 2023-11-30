import os, sys, subprocess, glob, random
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from copy import deepcopy
from .PDBfile_Converter import PDBImageFileToMols
from .Hydrocarbons_Builder import convex_bond_atom, Find_Vertex_v2
import numpy as np
from .Utility import Embedfrom2Dto3D, Plot_2Dmol, Plot_2Dmol_c
from .Heteroatom_Exchanger import *




##### Get the type of carbons ######
def mol_faH(mol):
    num_faH = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    ringinfo = mol.GetRingInfo()
    aring = ringinfo.AtomRings()
    faH_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() =='C' and atom.GetTotalNumHs() == 1 and atom.GetIsAromatic() \
            and atom.IsInRing() and len([r for r in aring if atom.GetIdx() in r and len(r) <= 6]) == 1]
    num_faH = len(faH_idx)
    return num_faH, faH_idx

def mol_faP(mol):
    num_faP = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ringinfo = mol.GetRingInfo()
    aring = [tmp for tmp in ringinfo.AtomRings() if len(tmp) <= 6]

    faP_idx1 = [b.GetBeginAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'C' and (b.GetEndAtom().GetSymbol() == 'O' or b.GetEndAtom().GetSymbol() == 'S') \
        and b.GetBeginAtom().GetIsAromatic() and (len([r for r in aring if b.GetEndAtomIdx() in r ]) == 0) and b.GetEndAtom().GetTotalNumHs() == 1 \
        and b.GetBondType() == Chem.rdchem.BondType.SINGLE] + \
        [b.GetEndAtomIdx() for b in bonds \
        if (b.GetBeginAtom().GetSymbol() == 'O' or b.GetBeginAtom().GetSymbol() == 'S') and b.GetEndAtom().GetSymbol() == 'C' \
        and b.GetEndAtom().GetIsAromatic() and (len([r for r in aring if b.GetBeginAtomIdx() in r ]) == 0) and b.GetBeginAtom().GetTotalNumHs() == 1 \
        and b.GetBondType() == Chem.rdchem.BondType.SINGLE]

    faP_idx2 = [b.GetBeginAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'C' and (b.GetEndAtom().GetSymbol() == 'O' or b.GetEndAtom().GetSymbol() == 'S') \
        and b.GetBeginAtom().GetIsAromatic() and (len([r for r in aring if b.GetEndAtomIdx() in r ]) == 0) and b.GetEndAtom().GetTotalNumHs() == 0 \
        and b.GetBondType() == Chem.rdchem.BondType.SINGLE] + \
        [b.GetEndAtomIdx() for b in bonds \
        if (b.GetBeginAtom().GetSymbol() == 'O' or b.GetBeginAtom().GetSymbol() == 'S') and b.GetEndAtom().GetSymbol() == 'C' \
        and b.GetEndAtom().GetIsAromatic() and (len([r for r in aring if b.GetBeginAtomIdx() in r ]) == 0) and b.GetBeginAtom().GetTotalNumHs() == 0 \
        and b.GetBondType() == Chem.rdchem.BondType.SINGLE]
        
    faP_idx3 = [b.GetBeginAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'C' and (b.GetEndAtom().GetSymbol() == 'O' or b.GetEndAtom().GetSymbol() == 'S') \
        and b.GetBeginAtom().GetIsAromatic() and not b.GetEndAtom().GetIsAromatic() and b.GetEndAtom().GetTotalNumHs() == 0 \
        and b.GetBondType() == Chem.rdchem.BondType.SINGLE] + \
        [b.GetEndAtomIdx() for b in bonds \
        if (b.GetBeginAtom().GetSymbol() == 'O' or b.GetBeginAtom().GetSymbol() == 'S') and b.GetEndAtom().GetSymbol() == 'C' \
        and not b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic() and b.GetBeginAtom().GetTotalNumHs() == 0 \
        and b.GetBondType() == Chem.rdchem.BondType.SINGLE]

    faP_idx4 = [b.GetBeginAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'C' and (b.GetEndAtom().GetSymbol() == 'O' or b.GetEndAtom().GetSymbol() == 'S') \
        and b.GetBeginAtom().GetIsAromatic() and not b.GetEndAtom().GetIsAromatic() and b.GetEndAtom().GetTotalNumHs() == 0 \
        and b.GetBondType() == Chem.rdchem.BondType.DOUBLE] + \
        [b.GetEndAtomIdx() for b in bonds \
        if (b.GetBeginAtom().GetSymbol() == 'O' or b.GetBeginAtom().GetSymbol() == 'S') and b.GetEndAtom().GetSymbol() == 'C' \
        and not b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic() and b.GetBeginAtom().GetTotalNumHs() == 0 \
        and b.GetBondType() == Chem.rdchem.BondType.DOUBLE]

    faP_idx = list(set(faP_idx1+faP_idx2+faP_idx3))
    num_faP = len(faP_idx)
    return num_faP, faP_idx

def mol_faS(mol):
    num_faS = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ringinfo = mol.GetRingInfo()

    aring = [tmp for tmp in ringinfo.AtomRings() if len(tmp) <= 6]

    faS_idx1 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() =='C' and atom.GetTotalNumHs() == 0 and atom.GetIsAromatic() \
            and len([n for n in atom.GetNeighbors() if (n.GetSymbol() == 'S') and (len([r for r in aring if n.GetIdx() in r ]) == 0) and not n.GetIsAromatic() ])==0 \
            and len([n for n in atom.GetNeighbors() if (n.GetSymbol() == 'O') and (len([r for r in aring if n.GetIdx() in r ]) == 0) and not n.GetIsAromatic() ])==0 \
            and len([r for r in aring if atom.GetIdx() in r ]) == 1 ]

    faS_idx2 = [b.GetBeginAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'C' and b.GetEndAtom().GetSymbol() == 'C' \
            and b.GetBeginAtom().GetIsAromatic() and not b.GetEndAtom().GetIsAromatic()] + \
        [b.GetEndAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'C' and b.GetEndAtom().GetSymbol() == 'C' \
            and not b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic()]

    faS_idx3 = [b.GetBeginAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'C' and b.GetEndAtom().GetSymbol() == 'N' \
        and b.GetBeginAtom().GetIsAromatic() and not b.GetEndAtom().GetIsAromatic() ] + \
        [b.GetEndAtomIdx() for b in bonds \
        if b.GetBeginAtom().GetSymbol() == 'N' and b.GetEndAtom().GetSymbol() == 'C' \
        and not b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic() ]

    faS_idx4 = [b.GetBeginAtomIdx() for b in bonds if b.GetBeginAtom().GetSymbol() == 'C' \
        and (b.GetEndAtom().GetSymbol() != 'O') and (b.GetEndAtom().GetSymbol() != 'S')\
        and b.GetBeginAtom().GetIsAromatic() and not b.GetEndAtom().GetIsAromatic() and b.GetEndAtom().GetTotalNumHs() == 1] + \
        [b.GetEndAtomIdx() for b in bonds if b.GetEndAtom().GetSymbol() == 'C' \
        and (b.GetBeginAtom().GetSymbol() != 'O') and (b.GetBeginAtom().GetSymbol() != 'S') \
        and not b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic() and b.GetBeginAtom().GetTotalNumHs() == 1]
    

    faS_idx = list(set(faS_idx1+faS_idx2+faS_idx3+faS_idx4))
    num_faS = len(faS_idx)
    return num_faS, faS_idx

def mol_faB(mol):
    num_faB = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    ringinfo = mol.GetRingInfo()
    aring = [tmp for tmp in ringinfo.AtomRings() if len(tmp) <= 6]

    faB_idx = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() =='C' and atom.GetIsAromatic() and atom.GetTotalNumHs() == 0 \
        and len([n.GetIdx() for n in atom.GetNeighbors() if n.GetIsAromatic()]) == 3 \
        #and sum([1 if b.GetIsAromatic() else 0 for b in atom.GetBonds()]) >= 2 \
        and len([r for r in aring if atom.GetIdx() in r]) >= 2 ]

    num_faB = len(faB_idx)
    return num_faB, faB_idx

def mol_faC(mol):
    num_faC = 0
    mol = AllChem.RemoveHs(mol)
    bonds = mol.GetBonds()
    atoms = mol.GetAtoms()

    #Aldehyde
    faC_idx1 = [bond.GetBeginAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE \
        and bond.GetBeginAtom().GetSymbol() == 'C' and bond.GetEndAtom().GetSymbol() == 'O' \
        and not bond.GetBeginAtom().IsInRing() and not bond.GetEndAtom().IsInRing() \
        and bond.GetBeginAtom().GetTotalNumHs() == 1 \
        and len([n.GetSymbol() for n in bond.GetBeginAtom().GetNeighbors() if n.GetSymbol() == 'C' or n.GetSymbol() == 'N' or n.GetSymbol() == 'S']) == 1]
    faC_idx2 = [bond.GetEndAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE \
        and bond.GetEndAtom().GetSymbol() == 'C' and bond.GetBeginAtom().GetSymbol() == 'O' \
        and not bond.GetEndAtom().IsInRing() and not bond.GetBeginAtom().IsInRing() \
        and bond.GetEndAtom().GetTotalNumHs() == 1 \
        and len([n.GetSymbol() for n in bond.GetEndAtom().GetNeighbors() if n.GetSymbol() == 'C' or n.GetSymbol() == 'N' or n.GetSymbol() == 'S']) == 1]

    #Ketone
    faC_idx3 = [bond.GetBeginAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE \
        and bond.GetBeginAtom().GetSymbol() == 'C' and bond.GetEndAtom().GetSymbol() == 'O' \
        and not bond.GetEndAtom().IsInRing() \
        and bond.GetBeginAtom().GetTotalNumHs() == 0 \
        and len([n.GetSymbol() for n in bond.GetBeginAtom().GetNeighbors() if n.GetSymbol() == 'C' or n.GetSymbol() == 'N' or n.GetSymbol() == 'S']) == 2]
    faC_idx4 = [bond.GetEndAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE \
        and bond.GetEndAtom().GetSymbol() == 'C' and bond.GetBeginAtom().GetSymbol() == 'O' \
        and not bond.GetBeginAtom().IsInRing() \
        and bond.GetEndAtom().GetTotalNumHs() == 0 \
        and len([n.GetSymbol() for n in bond.GetEndAtom().GetNeighbors() if n.GetSymbol() == 'C' or n.GetSymbol() == 'N' or n.GetSymbol() == 'S']) == 2]
    
    #Carboxylic Acid
    faC_idx5 = [bond.GetBeginAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE \
        and bond.GetBeginAtom().GetSymbol() == 'C' and bond.GetEndAtom().GetSymbol() == 'O' \
        and not bond.GetBeginAtom().GetIsAromatic() and not bond.GetEndAtom().IsInRing() \
        and len([n.GetSymbol() for n in bond.GetBeginAtom().GetNeighbors() if n.GetSymbol() == 'O']) == 2]
    faC_idx6 = [bond.GetEndAtom() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE \
        and bond.GetBeginAtom().GetSymbol() == 'O' and bond.GetEndAtom().GetSymbol() == 'C' \
        and not bond.GetEndAtom().GetIsAromatic() and not bond.GetBeginAtom().IsInRing() \
        and len([n.GetSymbol() for n in bond.GetEndAtom().GetNeighbors() if n.GetSymbol() == 'O']) == 2]

    faC_idx = list(set(faC_idx1+faC_idx2+faC_idx3+faC_idx4+faC_idx5+faC_idx6))
    num_faC = len(faC_idx)
    return num_faC, faC_idx

def mol_fal(mol):
    num_fal = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ringinfo = mol.GetRingInfo()
    aring = ringinfo.AtomRings()

    fal_idx1 = [bond.GetBeginAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
        and bond.GetBeginAtom().GetSymbol() == 'C' and not bond.GetBeginAtom().GetIsAromatic() and not bond.GetBeginAtom().IsInRing() \
        and bond.GetBeginAtom().GetTotalNumHs() == 1 \
        and bond.GetEndAtom().GetSymbol() == 'O' and not bond.GetEndAtom().GetIsAromatic() and not bond.GetEndAtom().IsInRing() \
        and bond.GetEndAtom().GetTotalNumHs() > 0 \
        and len([n.GetSymbol() for n in bond.GetBeginAtom().GetNeighbors() if n.GetSymbol() == 'C']) == 1] \
        + [bond.GetEndAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
        and bond.GetEndAtom().GetSymbol() == 'C' and not bond.GetEndAtom().GetIsAromatic() and not bond.GetEndAtom().IsInRing() \
        and bond.GetEndAtom().GetTotalNumHs() == 1 \
        and bond.GetBeginAtom().GetSymbol() == 'O' and not bond.GetBeginAtom().GetIsAromatic() and not bond.GetBeginAtom().IsInRing() \
        and bond.GetBeginAtom().GetTotalNumHs() > 0 \
        and len([n.GetSymbol() for n in bond.GetEndAtom().GetNeighbors() if n.GetSymbol() == 'C']) == 1]

    fal_idx2 = [bond.GetBeginAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
        and bond.GetBeginAtom().GetSymbol() == 'C' and not bond.GetBeginAtom().GetIsAromatic() and not bond.GetBeginAtom().IsInRing()\
        and bond.GetBeginAtom().GetTotalNumHs() >= 2 ]\
        + [bond.GetEndAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
        and bond.GetEndAtom().GetSymbol() == 'C' and not bond.GetEndAtom().GetIsAromatic() and not bond.GetEndAtom().IsInRing()\
        and bond.GetEndAtom().GetTotalNumHs() >= 2 ]

    fal_idx3 = [bond.GetBeginAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
        and bond.GetBeginAtom().GetSymbol() == 'C' and not bond.GetBeginAtom().GetIsAromatic() and bond.GetBeginAtom().IsInRing()\
        and bond.GetBeginAtom().GetTotalNumHs() >= 1 ]\
        + [bond.GetEndAtomIdx() for bond in bonds \
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
        and bond.GetEndAtom().GetSymbol() == 'C' and not bond.GetEndAtom().GetIsAromatic() and bond.GetEndAtom().IsInRing()\
        and bond.GetEndAtom().GetTotalNumHs() >= 1 ]

    fal_idx4 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 1 \
        and 'N' in [n.GetSymbol() for n in atom.GetNeighbors()] and len([n.GetSymbol() for n in atom.GetNeighbors()]) == 3] \
        + [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() <= 2 \
        and 'O' in [n.GetSymbol() for n in atom.GetNeighbors() if n.IsInRing()] ]

    fal_idx5 =[atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() =='C' and atom.IsInRing() and not atom.GetIsAromatic() and atom.GetTotalNumHs() == 0 \
        and sum([1 if b.GetIsAromatic() else 0 for b in atom.GetBonds()]) < 3 \
        and len([r for r in aring if atom.GetIdx() in r]) > 1 ]

    fal_idx6 = [atom.GetIdx() for atom in atoms \
        if not atom.GetIsAromatic() and atom.GetSymbol() == 'C' and atom.GetTotalNumHs()==0 \
        and len([n for n in atom.GetNeighbors() if (n.GetSymbol() =='C' or n.GetSymbol() =='S') and not n.IsInRing()]) == 1 \
        and len([n for n in atom.GetNeighbors() if (n.GetSymbol() =='O') and (not n.IsInRing()) and (mol.GetBondBetweenAtoms(atom.GetIdx(),n.GetIdx()).GetBondType()==Chem.rdchem.BondType.DOUBLE)]) == 0]

    fal_idx7 = [atom.GetIdx() for atom in atoms \
        if not atom.GetIsAromatic() and atom.GetSymbol() == 'C' and atom.GetTotalNumHs() == 0 \
        and len([n for n in atom.GetNeighbors() if n.GetSymbol() =='O' and not n.IsInRing() and len(n.GetNeighbors())<=2]) == 1 \
        and len([b.GetBeginAtom() for b in atom.GetBonds() if (b.GetBeginAtom().GetSymbol()=='O' or b.GetEndAtom().GetSymbol()=='O') and b.GetBondTypeAsDouble() == 1])==1]
    
    fal_idx8 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 1 \
        and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol()=='C']) == 2]

    fal_idx9 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 0 \
        and len([n.GetSymbol() for n in atom.GetNeighbors()]) == 4]
    
    fal_idx9_1 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 0 \
        and len([n.GetSymbol() for n in atom.GetNeighbors()]) == 3]

    fal_idx9_2 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and (not atom.IsInRing()) and atom.GetTotalNumHs() == 0 \
        and len([n.GetSymbol() for n in atom.GetNeighbors()]) == 4]

    fal_idx10 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() <= 1 \
            and len([n for n in atom.GetNeighbors()]) == 3]

    fal_idx11 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 2 \
            and len([n for n in atom.GetNeighbors()]) == 1]

    fal_idx = fal_idx1 + fal_idx2 + fal_idx3 + fal_idx4 + fal_idx5 + fal_idx6 + fal_idx7 + fal_idx8 + fal_idx9 + fal_idx9_1 + fal_idx9_2 + fal_idx10 + fal_idx11
    fal_idx = list(set(fal_idx))
    num_faC, faC_idx = mol_faC(mol)
    fal_idx = [idx for idx in fal_idx if idx not in faC_idx]

    num_fal = len(fal_idx)
    return num_fal, fal_idx


### Need to be checked
def mol_falas(mol): #CH3 or nonprotonated
    num_fal = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ringinfo = mol.GetRingInfo()
    aring = ringinfo.AtomRings()

    fal_idx1 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 3] 
    fal_idx2 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 0 \
            and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol() == 'O']) < 2]
    fal_idx3 =[atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() =='C' and atom.IsInRing() and not atom.GetIsAromatic() and atom.GetTotalNumHs() == 0 \
            and sum([1 if b.GetIsAromatic() else 0 for b in atom.GetBonds()]) < 3 \
            and len([r for r in aring if atom.GetIdx() in r]) > 1 ]
    fal_idx4 = [atom.GetIdx() for atom in atoms \
        if not atom.GetIsAromatic() and atom.GetSymbol() == 'C' and atom.GetTotalNumHs() == 0 \
            and len([n for n in atom.GetNeighbors() if n.GetSymbol() =='O' and not n.IsInRing() and len(n.GetNeighbors())<=2]) == 1 \
            and len([b.GetBeginAtom() for b in atom.GetBonds() if (b.GetBeginAtom().GetSymbol()=='O' or b.GetEndAtom().GetSymbol()=='O') and b.GetBondTypeAsDouble() == 1])==1]
    fal_idx5 = [atom.GetIdx() for atom in atoms \
        if not atom.GetIsAromatic() and atom.GetSymbol() == 'C' and atom.GetTotalNumHs()==0 \
            and len([n for n in atom.GetNeighbors() if (n.GetSymbol() =='C' or n.GetSymbol() =='S') and not n.IsInRing()]) == 1]
    fal_idx6 = [atom.GetIdx() for atom in atoms \
        if not atom.GetIsAromatic() and atom.GetSymbol() == 'C' and atom.GetTotalNumHs()==0 and atom.IsInRing() \
            and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol() == 'O']) >= 1 ]
    fal_idx7 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 0 \
            and len([n for n in atom.GetNeighbors()]) == 4]
    fal_idx8 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 0 \
            and len([n for n in atom.GetNeighbors()]) == 3]
    fal_idx9 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 0 \
            and len([n for n in atom.GetNeighbors()]) == 3]
    fal_idx10 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 0 \
            and len([n for n in atom.GetNeighbors()]) == 4]
    
    # Check
    fal_idx11= [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 0]

    fal_idx = fal_idx1 + fal_idx2 + fal_idx3 + fal_idx4 + fal_idx5 + fal_idx6 + fal_idx7 + fal_idx8 +fal_idx9 + fal_idx10 + fal_idx11
    fal_idx = list(set(fal_idx))
    num_faC, faC_idx = mol_faC(mol)
    fal_idx = [idx for idx in fal_idx if idx not in faC_idx]

    num_fal = len(fal_idx)
    return num_fal, fal_idx

def mol_falH(mol): #CH or CH2
    num_fal = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ringinfo = mol.GetRingInfo()
    aring = [tmp for tmp in ringinfo.AtomRings() if len(tmp) <= 6]
    
    fal_idx0 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 1 \
            and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol()=='O']) == 0 \
            and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol()=='C']) == 1]
    fal_idx1 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 2]
    fal_idx2 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 1 \
            and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol()=='O']) == 0 \
            and len([n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol()=='C']) == 2] 
    fal_idx3 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 2 \
            and len([r for r in aring if atom.GetIdx() in r]) >= 1 ]
    fal_idx4 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() == 1 \
            and len([r for r in aring if atom.GetIdx() in r]) >= 1 ]
    fal_idx5 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and atom.GetTotalNumHs() == 2 \
            and len([r for r in aring if atom.GetIdx() in r]) < 1 ]
    fal_idx6 = [atom.GetIdx() for atom in atoms \
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and not atom.IsInRing() and atom.GetTotalNumHs() == 2 \
            and len([n for n in atom.GetNeighbors()]) == 1]
    fal_idx7 = [atom.GetIdx() for atom in atoms \
        if not atom.GetIsAromatic() and atom.GetSymbol() == 'C' and atom.GetTotalNumHs() == 1 \
        and len([n for n in atom.GetNeighbors() if n.GetSymbol() =='O' and not n.IsInRing() and len(n.GetNeighbors())<=2]) == 1 \
        and len([b.GetBeginAtom() for b in atom.GetBonds() if (b.GetBeginAtom().GetSymbol()=='O' or b.GetEndAtom().GetSymbol()=='O') and b.GetBondTypeAsDouble() == 1])==1]

    fal_idx = fal_idx0 + fal_idx1 + fal_idx2 + fal_idx3 + fal_idx4 + fal_idx5 + fal_idx6 + fal_idx7
    fal_idx = list(set(fal_idx))
    num_fal = len(fal_idx)
    return num_fal, fal_idx

def mol_falO(mol):
    num_fal = 0
    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ringinfo = mol.GetRingInfo()
    aring = ringinfo.AtomRings()

    num_fal, fal_idx = mol_fal(mol)
    num_faC, faC_idx = mol_faC(mol)
    falO_idx1 = [idx for idx in fal_idx \
                if len([n.GetSymbol() for n in atoms[idx].GetNeighbors() if n.GetSymbol()=='O']) >= 1 \
                and idx not in faC_idx ]

    fal_idx = falO_idx1
    fal_idx = list(set(fal_idx))
    num_fal = len(fal_idx)
    return num_fal, fal_idx

####################################

##### Calculate 13C-NMR parameters ######

def carbon_nmr(mols, draw=False):

    faH = 0
    faP = 0
    faS = 0
    faB = 0
    faC = 0

    fal = 0
    falas = 0
    falH = 0
    falO = 0

    total_c = 0

    for i, mol in enumerate(mols):
        atoms = mol.GetAtoms()
        num_c = len([atom for atom in atoms if atom.GetSymbol()=='C'])
        num_faH, faH_idx = mol_faH(mol)
        num_faP, faP_idx = mol_faP(mol)
        num_faS, faS_idx = mol_faS(mol)
        num_faB, faB_idx = mol_faB(mol)
        num_faC, faC_idx = mol_faC(mol)

        faH += num_faH
        faP += num_faP
        faS += num_faS
        faB += num_faB
        faC += num_faC

        num_fal, fal_idx = mol_fal(mol)
        num_falas, falas_idx = mol_falas(mol)
        num_falH, falH_idx = mol_falH(mol)
        num_falO, falO_idx = mol_falO(mol)
        fal += num_fal
        falas += num_falas
        falH  += num_falH
        falO  += num_falO

        total_c += num_c
        test = sum([num_faH, num_faP, num_faS, num_faB, num_faC, num_fal])
        #print(num_faH, num_faP, num_faS, num_faB, num_faC, num_fal, num_falas, num_falH, num_falO, test, num_c)
        if draw:
            carbon_nmr_individual_highlight(mol,'2d_%d.png' % (i+1))
        if test != num_c:
            print(num_faH, num_faP, num_faS, num_faB, num_faC, num_fal, num_falas, num_falH, num_falO, test, num_c)
            Plot_2Dmol(mol, ha=faH_idx+faP_idx+faS_idx+faB_idx+faC_idx+fal_idx)
            check_idx = [atom.GetIdx() for atom in atoms if atom.GetIsAromatic() and atom.GetSymbol() == 'C']
            Plot_2Dmol(mol, ha=fal_idx)
            #Plot_2Dmol(mol, ha=faC_idx)
            mol3D = Embedfrom2Dto3D(mol)
            pdbfile = open('check_structure.pdb','w')
            pdbfile.write(AllChem.MolToPDBBlock(mol3D))
            pdbfile.close()
            #Plot_2Dmol(mol, ha=check_idx)
            #Plot_2Dmol(mol, ha=fal_idx)
            sys.exit()

    faN = (faP + faS + faB)
    fapr = faN + faH

    result = {'fa':(fapr+faC)/total_c,
              'fapr':fapr/total_c,
              'faN':faN/total_c,
              'faH':faH/total_c,
              'faP':faP/total_c,
              'faS':faS/total_c,
              'faB':faB/total_c,
              'faC':faC/total_c,
              'fal':fal/total_c,
              'fal*':falas/total_c,
              'falH':falH/total_c,
              'falO':falO/total_c,
              'num_C':total_c,
              }
    return result


def carbon_nmr_individual(mol):

    atoms = mol.GetAtoms()
    num_c = len([atom for atom in atoms if atom.GetSymbol()=='C'])
    num_faH, faH_idx = mol_faH(mol)
    num_faP, faP_idx = mol_faP(mol)
    num_faS, faS_idx = mol_faS(mol)
    num_faB, faB_idx = mol_faB(mol)
    num_faC, faC_idx = mol_faC(mol)
    num_fal, fal_idx = mol_fal(mol)
    faH = num_faH
    faP = num_faP
    faS = num_faS
    faB = num_faB
    faC = num_faC
    faN = faP + faS + faB
    fapr = faN + faH

    num_fal, fal_idx = mol_fal(mol)
    num_falas, falas_idx = mol_falas(mol)
    num_falH, falH_idx = mol_falH(mol)
    num_falO, falO_idx = mol_falO(mol)
    fal = num_fal
    falas = num_falas
    falH  = num_falH
    falO  = num_falO

    fal = num_fal
    total_c = num_c
    test = sum([num_faH, num_faP, num_faS, num_faB, num_faC, num_fal])
    result = {'fapr':fapr,
              'faN':faN,
              'faH':faH,
              'faP':faP,
              'faS':faS,
              'faB':faB,
              'faC':faC,
              'fal':fal,
              'fal*':falas,
              'falH':falH,
              'falO':falO,
              'num_C':total_c,
              }

    result_ratio = {'fapr':fapr/total_c,
                    'faN':faN/total_c,
                    'faH':faH/total_c,
                    'faP':faP/total_c,
                    'faS':faS/total_c,
                    'faB':faB/total_c,
                    'faC':faC/total_c,
                    'fal':fal/total_c,
                    'fal*':falas/total_c,
                    'falH':falH/total_c,
                    'falO':falO/total_c,
                    'num_C':total_c,
                    }
    return result_ratio

def carbon_nmr_individual_highlight(mol, pngfile):

    atoms = mol.GetAtoms()
    num_c = len([atom for atom in atoms if atom.GetSymbol()=='C'])
    num_faH, faH_idx = mol_faH(mol)
    num_faP, faP_idx = mol_faP(mol)
    num_faS, faS_idx = mol_faS(mol)
    num_faB, faB_idx = mol_faB(mol)
    num_faC, faC_idx = mol_faC(mol)
    num_fal, fal_idx = mol_fal(mol)
    faH = num_faH
    faP = num_faP
    faS = num_faS
    faB = num_faB
    faC = num_faC
    faN = faP + faS + faB
    fapr = faN + faH

    num_fal, fal_idx = mol_fal(mol)
    num_falas, falas_idx = mol_falas(mol)
    num_falH, falH_idx = mol_falH(mol)
    num_falO, falO_idx = mol_falO(mol)
    fal = num_fal
    falas = num_falas
    falH  = num_falH
    falO  = num_falO
    total_c = num_c

    test = sum([num_faH, num_faP, num_faS, num_faB, num_faC, num_fal])
    test2 = sum([falas, falH])
    print(num_faH, num_faP, num_faS, num_faB, num_faC, num_fal, test, num_c)
    print(num_fal, falas, falH, falO)
    colors = [(1.0,0.0,0.0),
              (0.0,1.0,0.0),
              (0.0,0.0,1.0),
              (1.0,0.0,1.0),
              (1.0,1.0,0.0),
              (0.0,1.0,1.0),
              (1.0,1.0,1.0),
              (0.5,0.5,0.0),
              (0.2,0.5,0.5),]
    atom_color = {}
    new_color = []
    for i, at in enumerate(faH_idx):
        atom_color[at] = colors[0]
        new_color.append(colors[0])
    for i, at in enumerate(faP_idx):
        atom_color[at] = colors[1]
        new_color.append(colors[1])
    for i, at in enumerate(faS_idx):
        atom_color[at] = colors[2]
        new_color.append(colors[2])
    for i, at in enumerate(faB_idx):
        atom_color[at] = colors[3]
        new_color.append(colors[3])
    for i, at in enumerate(faC_idx):
        atom_color[at] = colors[4]
        new_color.append(colors[4])
    for i, at in enumerate(fal_idx):
        atom_color[at] = colors[5]
        new_color.append(colors[5])
    for i, at in enumerate(falas_idx):
        atom_color[at] = colors[6]
        new_color.append(colors[6])
    for i, at in enumerate(falH_idx):
        atom_color[at] = colors[7]
        new_color.append(colors[7])
    for i, at in enumerate(falO_idx):
        atom_color[at] = colors[8]
        new_color.append(colors[8])

    hit_ats = faH_idx + faS_idx + faP_idx + faB_idx + faC_idx + fal_idx
    AllChem.Compute2DCoords(mol, nFlipsPerSample=1)
    Chem.rdCoordGen.AddCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(800,1000)
    drawer.DrawMolecule(mol,highlightAtoms=hit_ats,
                            highlightAtomColors=atom_color,
                            highlightBonds=[],
                            highlightBondColors={})
    drawer.FinishDrawing()
    #drawer.WriteDrawingText(pngfile)
    
    #svg = drawer.GetDrawingText().replace('svg:','')
    #if num_fal != falas + falH:
    if True:
        Plot_2Dmol(mol, ha=fal_idx, pngfilename=pngfile)
        Plot_2Dmol(mol, ha=falO_idx, pngfilename=pngfile)
        
    result = {'faH':faH/total_c,
              'faP':faP/total_c,
              'faS':faS/total_c,
              'faB':faB/total_c,
              'faC':faC/total_c,
              'fal':fal/total_c}
    result_idx = {'faH':faH_idx,
                  'faP':faP_idx,
                  'faS':faS_idx,
                  'faB':faB_idx,'faC':faC_idx,'fal':fal_idx}
    return result, result_idx


def carbon_nmr_individual_test(mol):

    atoms = mol.GetAtoms()
    num_c = len([atom for atom in atoms if atom.GetSymbol()=='C'])
    num_faH, faH_idx = mol_faH(mol)
    num_faP, faP_idx = mol_faP(mol)
    num_faS, faS_idx = mol_faS(mol)
    num_faB, faB_idx = mol_faB(mol)
    num_faC, faC_idx = mol_faC(mol)
    num_fal, fal_idx = mol_fal(mol)
    faH = num_faH
    faP = num_faP
    faS = num_faS
    faB = num_faB
    faC = num_faC

    num_fal, fal_idx = mol_fal(mol)
    num_falas, falas_idx = mol_falas(mol)
    num_falH, falH_idx = mol_falH(mol)
    num_falO, falO_idx = mol_falO(mol)
    fal = num_fal
    falas = num_falas
    falH  = num_falH
    falO  = num_falO

    fal = num_fal
    total_c = num_c
    test = sum([num_faH, num_faP, num_faS, num_faB, num_faC, num_falH,num_falas])
    print('faH:%d faP:%d faS:%d faB:%d faC:%d fal:%d falH:%d fal*:%d falO:%d sum:%d total_c:%d' %(num_faH, num_faP, num_faS, num_faB, num_faC, num_fal, num_falH, falas, falO, test, num_c))
    #if test != num_c:
    check_idx = [atom.GetIdx() for atom in atoms if atom.GetIsAromatic()]
    #Plot_2Dmol(mol, ha=faH_idx+faP_idx+faS_idx+faB_idx+faC_idx+falH_idx+falas_idx)
    Plot_2Dmol(mol, ha=falas_idx)
    result = {'faH':faH/total_c,'faP':faP/total_c,'faS':faS/total_c,'faB':faB/total_c,'faC':faC/total_c,'fal':fal/total_c}
    return result


##### Calculate 13C-NMR parameters ######
#########################################


##### Calculate Elemental information ######

def current_element(mols):

    total_c = 0
    total_h = 0
    total_n = 0
    total_o = 0
    total_s = 0
    total_atoms = 0

    for i, mol in enumerate(mols):
        #mol_file, no H added 
        m1 = mol
        m1 = AllChem.RemoveHs(m1)
        chem =   [atom for atom in m1.GetAtoms()]
        chem_h = [atom.GetTotalNumHs() for atom in m1.GetAtoms()]
        chem_c = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'C']
        chem_n = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'N']
        chem_o = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'O']
        chem_s = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'S']

        total_c += len(chem_c)
        total_h += sum(chem_h)
        total_n += len(chem_n)
        total_o += len(chem_o)
        total_s += len(chem_s)

    total_atoms = sum([total_c,total_h,total_n,total_o,total_s])

    result = {'C':100*total_c/total_atoms,'H':100*total_h/total_atoms,'N':100*total_n/total_atoms,
            'O':100*total_o/total_atoms,'S':100*total_s/total_atoms}
    return result

def heteroatoms_details(mols):
	mols2 = []
	ether_o = 0
	phenolic_o = 0
	carbonyl_o = 0
	cross_o_1 = 0
	cross_o_2 = 0

	pyridine_n = 0
	pyrrolic_n = 0
	quatern_n = 0

	thiophene_s = 0
	cross_s_1 = 0
	cross_s_2 = 0	

	for i, mol in enumerate(mols):
		phenolic_o_idx = num_phenolic_o(mol)
		ether_o_idx = num_ether_o(mol)
		carbonyl_o_idx = num_carbonyl_o(mol)

		#print('O',len(phenolic_o_idx), len(ether_o_idx), len(carbonyl_o_idx))
		
		pyridine_n_idx = num_pyridine_n(mol)
		pyrrolic_n_idx = num_pyrrolic_n(mol)
		quatern_n_idx = num_quatern_n(mol)

		#print('N',len(pyridine_n_idx), len(pyrrolic_n_idx), len(quatern_n_idx))
		mol2 = deepcopy(mol)
		mol2, success = Convert_OH2CH3(mol2)
		#Plot_2Dmol(mol)
		#Plot_2Dmol(mol2)
		
		for j in range(40):
			mol2, success = Convert_Aromatic2Aliphatic(mol2)

		for j in range(40):
			success, mol2 = Heteroatom_Func_OH(mol2)
		#print(i, success)
		#atoms1 = mol.GetAtoms()
		#atoms2 = mol2.GetAtoms()
		#print(sum([atom.GetTotalNumHs() for atom in atoms1])-sum([atom.GetTotalNumHs() for atom in atoms2]))
		#Plot_2Dmol(mol)
		#Plot_2Dmol(mol2)
		#sys.exit()
		mols2.append(mol2)

	return mols2

##### Calculate Elemental information ######


def Bridgehead_count(m):
    carbon_CH = 0
    carbon_B = 0
    atoms = m.GetAtoms()
    check = [[n.GetSymbol() for n in atom.GetNeighbors()] for atom in atoms]
    #print(check)
    ring = m.GetRingInfo()
    aring = ring.AtomRings()
    aring_fl = []
    [[aring_fl.append(ari) for ari in ar] for ar in aring]
    
    for atom in atoms:
        aind = atom.GetIdx()
        count = aring_fl.count(aind)
        if count > 0 and atom.GetSymbol() != 'H':
            asym = [n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol() != 'H']
            #print(atom.GetSymbol(), asym, count)
            if len(asym) == 2:
                carbon_CH += 1
            elif len(asym) == 3:
                carbon_B += 1
    symbols = [atom.GetSymbol() for atom in atoms if atom.GetSymbol() != 'H']
    carbon_CH_r = carbon_CH/len(symbols)
    carbon_B_r = carbon_B/len(symbols)
    #print(carbon_CH_r, carbon_B_r)

    return carbon_CH, carbon_B, len(symbols)


def mean_CHratio1(mol_list):

    bv = []
    for k, mol in enumerate(mol_list):
        carbon_CH, carbon_B, total_C = Bridgehead_count(mol)
        bv_val = carbon_CH / total_C
        bv.append(bv_val)
    
    return np.mean(bv)

def mean_CHratio(mols_list):

    mols_bv = []
    for j, mol_list in enumerate(mols_list):
        bv = []
        for k, mol in enumerate(mol_list):
            carbon_CH, carbon_B, total_C = Bridgehead_count(mol)
            bv_val = carbon_CH / total_C
            bv.append(bv_val)
        mols_bv.append(np.mean(bv))
        
    return np.mean(mols_bv)

def examine_all(mols):
	total_cs = 0
	aromaticb_cs = 0
	aromaticbh_cs = 0
	aliphatich_cs = 0
	fah_cs = 0
	ali_add = 0

	total_o = 0
	total_h = 0
	total_s = 0
	total_n = 0
	total_c = 0

	for i, mol in enumerate(mols):
		#mol_file, no H added
		m1 = AllChem.RemoveHs(mol)
		chem =   [atom for atom in m1.GetAtoms()]
		chem_h = [atom.GetTotalNumHs() for atom in m1.GetAtoms()]
		chem_c = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'C']
		chem_n = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'N']
		chem_o = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'O']
		chem_s = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'S']

		total_h += sum(chem_h)
		total_c += len(chem_c)
		total_n += len(chem_n)
		total_o += len(chem_o)
		total_s += len(chem_s)
		
		chem_caro = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() <= 1]
		chem_caro_b  = [atom for atom in chem_caro if len([n for n in atom.GetNeighbors() if n.IsInRing()]) < 3]
		chem_caro_bh = [atom for atom in chem_caro if len([n for n in atom.GetNeighbors() if n.IsInRing()]) == 3]
		chem_cali_h = [atom for atom in m1.GetAtoms() if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 1]

		chem_caro_b_fah = [atom for atom in chem_caro_b if atom.GetTotalNumHs() == 1]
		chem_caro_b_fas = [atom for atom in chem_caro_b if atom.GetTotalNumHs() == 0 and len([n for n in atom.GetNeighbors()]) == 3]
		chem_caro_b_fap = [atom for atom in chem_caro_b if atom.GetTotalNumHs() == 0 and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'O']) > 0]
 
		#aro_cb = len(chem_caro) - len(chem_caro_bh)
		aro_cb = len(chem_caro_b)
		aro_cbh = len(chem_caro_bh)
		ali_c = len(chem_cali_h)
		aromaticb_cs += aro_cb
		aromaticbh_cs += aro_cbh
		aliphatich_cs += ali_c
	#print(aromaticb_cs / total_c, aromaticbh_cs / total_c, aliphatich_cs / total_c)
	#print(aromaticb_cs / (aromaticb_cs+aromaticbh_cs), aromaticbh_cs / (aromaticb_cs+aromaticbh_cs))
	#print(total_c,total_h)
	aromatic_cs = aromaticbh_cs + aromaticb_cs
	total_atoms = sum([total_h, total_c, total_n, total_o, total_s])
	calculate_ratio = {'C':100*total_c/total_atoms,
	                   'N':100*total_n/total_atoms, 
					   'H':100*total_h/total_atoms, 
					   'S':100*total_s/total_atoms, 
					   'O':100*total_o/total_atoms}
	result ={'Num_C': total_c, 'Num_H': total_h, 'Num_N':total_n, 'Num_O':total_o, 'Num_S':total_s, 
	'Aromatic_BH_C':aromaticbh_cs/aromatic_cs, 'Aromatic_B_C':aromaticb_cs/aromatic_cs,
	'Aromatic_BH_C_ratio':aromaticbh_cs/total_c, 'Aromatic_B_C_ratio':aromaticb_cs/total_c,
	'Aliphatic_C_ratio':aliphatich_cs/total_c, 'Num_atoms':total_atoms, 'current_ratio':calculate_ratio,
	}

	return result


#Should be collected
def calculate_element_needed(result, element_ratio_each):

	#element_ratio_norm = [e / element_ratio_each[0] for e in element_ratio_each]
	total_c = result['Num_C']
	total_h = result['Num_H']
	total_n = result['Num_N']
	total_o = result['Num_O']
	total_s = result['Num_S']
	total_atoms = result['Num_atoms']
	element_ratio = []
	for key, value in element_ratio_each.items():
		element_ratio.append(value / element_ratio_each['C'])

	target_nums = [e*total_c for e in element_ratio]
	element_seq = ['C','N','H','S','O']
	target_nums_d = {}
	for e, num in zip(element_seq,target_nums):
		#target_nums_d[e]=round(num-result['Num_'+e])
		target_nums_d[e]=round(num)
	return target_nums_d


def solubility_equation(fa,H,C,N,O,S):

    a = (7.0 + 63.5*fa + 63.5*(H/C) + 106*(O/C) + 51.8*(N+S)/C)
    b = (-10.9 + 12*fa + 13.9*(H/C) + 5.5*(O/C) - 2.8*(N+S)/C)
    delta = a / b

    #unit (cal/cm^3)^0.5
    return delta