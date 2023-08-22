from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from .Hydrocarbons_Builder import convex_bond_atom
from ase.io import read, write
from ase import Atoms, Atom
from copy import deepcopy
import numpy as np
import os, sys, glob, subprocess
from ase.visualize import view
from .Utility import Embedfrom2Dto3D, Plot_2Dmol, Plot_2Dmol_tmp


def MOLImageFileToMols(mol_filename):
    f = open(mol_filename,'r')
    lines = f.readlines()
    end_index = [l for l in range(len(lines)) if 'M  END' in lines[l]]
    start = 0

    ms = []
    for i in range(len(end_index)):
        end = end_index[i] + 1
        mol_block = lines[start:end]
        start = end
        mol_block_str = "".join(mol_block)
        m = AllChem.MolFromMolBlock(mol_block_str,removeHs=False)
        #print("test")
        #m = AllChem.RemoveHs(m, updateExplicitCount=True)
        ms.append(m)
    return ms

def MolsToMOLImageFile(mols,mol_filename):
    f = open(mol_filename,'w')
    for i, mol in enumerate(mols):
        mol3D = Embedfrom2Dto3D(mol)
        pdbblock = AllChem.MolToMolBlock(mol3D)
        f.write(pdbblock)
        f.flush()
    f.close()
    
    return mol_filename
