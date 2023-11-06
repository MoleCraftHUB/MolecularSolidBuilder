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

#Up-to-data

def PDBImageFileToMols(pdb_filename):
    f = open(pdb_filename,'r')
    lines = f.readlines()
    end_index = [l for l in range(len(lines)) if 'END' in lines[l]]
    start = 0

    ms = []
    for i in range(len(end_index)):
        end = end_index[i] + 1
        pdb_block = lines[start:end]
        start = end
        pdb_block_str = "".join(pdb_block)
        m = AllChem.MolFromPDBBlock(pdb_block_str,removeHs=False)
        #print("test")
        #m = AllChem.RemoveHs(m, updateExplicitCount=True)
        ms.append(m)
    return ms

def MolsToPDBImageFile(mols,pdb_filename):
    f = open(pdb_filename,'w')
    for i, mol in enumerate(mols):
        mol3D = Embedfrom2Dto3D(mol)
        pdbblock = AllChem.MolToPDBBlock(mol3D,flavor=1)
        f.write(pdbblock)
        f.flush()
    f.close()
    
    return pdb_filename
