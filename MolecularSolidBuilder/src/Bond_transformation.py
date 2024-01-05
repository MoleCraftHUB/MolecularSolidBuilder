import os, sys, subprocess, glob, random
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def Convert_Double2SingleBond(mol, bond_index):
    AllChem.Kekulize(mol)


    mol = AllChem.RemoveHs(mol)
    return

def Convert_Single2DoubleBond(mol, bond_index):
    AllChem.Kekulize(mol)


    mol = AllChem.RemoveHs(mol)
    return


def Select_Double2SingleBond(mol):
    AllChem.Kekulize(mol)


    mol = AllChem.RemoveHs(mol)
    return

def Select_Single2DoubleBond(mol):
    AllChem.Kekulize(mol)


    mol = AllChem.RemoveHs(mol)
    return