from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from Build_HC_revise import convex_bond_atom
from ase.io import read, write
from ase import Atoms, Atom
from copy import deepcopy
import numpy as np
import os, sys, glob, subprocess
from ase.visualize import view

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