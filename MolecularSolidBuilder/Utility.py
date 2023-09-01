import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase.io import read, write
import numpy as np
import numpy.linalg as LA
import os, sys, glob, subprocess
from itertools import combinations
from copy import deepcopy
import itertools
from .MinimumBoundingBox3D import Get3DMinimumBoundingBox
from MinimumBoundingBox import MinimumBoundingBox
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.mplot3d import Axes3D

def SMItoPDBblock(smi,mmb=False):
	mol = AllChem.MolFromSmiles(smi)
	mol_3d = Embedfrom2Dto3D(mol)
	pdb_string = AllChem.MolToPDBBlock(mol_3d)
	if mmb:
		box, pdb_string = Get3DMinimumBoundingBox(pdb_string)
	return pdb_string

def SMItoPDBFile(smi,fname,mmb=False):
	pdb_string = SMItoPDBblock(smi)
	pdb_file = open(fname,'w')
	pdb_file.write(pdb_string)
	pdb_file.close()

	return fname




def PAH_screen1(mol):

    mol = AllChem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    ringinfo = mol.GetRingInfo()
    #aring = ringinfo.AtomRings()
    aring = [r for r in ringinfo.AtomRings() if len(r) == 5 \
            and len([rr for rr in r if atoms[rr].GetTotalNumHs() >= 1]) > 1]

    aring_five_flatt = []
    [[aring_five_flatt.append(ri) for ri in r] for r in ringinfo.AtomRings() if len(r) == 5]
    aring_five_flatt_set = list(set(aring_five_flatt))

    if len(aring) == 0 and len(aring_five_flatt_set) == len(aring_five_flatt):
        return True
    else:
        return False


def Plot_2Dmol_InNoteBook(mol, ha=None, ca=None):
    
    from rdkit.Chem.Draw import IPythonConsole
    from IPython.display import SVG

    check1 = deepcopy(mol)
    check1 = AllChem.RemoveHs(check1)
    AllChem.Compute2DCoords(check1, nFlipsPerSample=1)
    Chem.rdCoordGen.AddCoords(check1)
    drawer = rdMolDraw2D.MolDraw2DSVG(800,500)
    drawer.DrawMolecule(check1, highlightAtoms=ha)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    

    return svg


def Plot_2Dmol_c(mol, ha=None, ca=None, pngfilename='2d.png'):
    
    check1 = deepcopy(mol)
    check1 = AllChem.RemoveHs(check1)
    AllChem.Compute2DCoords(check1, nFlipsPerSample=2)
    #Chem.rdCoordGen.AddCoords(check1)
    Chem.Draw.MolToFile(check1,pngfilename, size=(2000,2000), highlightAtoms=ha)
    subprocess.call('imgcat %s' % pngfilename ,shell=True)

    return pngfilename

def Plot_2Dmol_tmp(mol, ha=None, pngfilename='2d.png'):
    
    check1 = deepcopy(mol)
    check1 = AllChem.AddHs(check1)
    #check1 = AllChem.RemoveHs(check1)
    AllChem.Compute2DCoords(check1, nFlipsPerSample=2)
    #Chem.rdCoordGen.AddCoords(check1)
    #Chem.Draw.MolToFile(check1,'./test.png' , size=(200,200),kekulize=True,highlightAtoms=avail_atom_idx_1)#, highlightBonds=convex_bond)
    if ha != None:
        Chem.Draw.MolToFile(check1,pngfilename, size=(500,500), highlightAtoms=ha)
    else:
        Chem.Draw.MolToFile(check1,pngfilename, size=(500,500))
    subprocess.call('imgcat %s' % pngfilename ,shell=True)

    return pngfilename


def Plot_2Dmol(mol, ha=None, pngfilename='2d.png',size=(1000,1000)):
    
    check1 = deepcopy(mol)
    #check1 = AllChem.RemoveHs(check1)
    AllChem.Compute2DCoords(check1, nFlipsPerSample=10,bondLength=True)
    #Chem.rdCoordGen.AddCoords(check1)
    if ha != None:
        Chem.Draw.MolToFile(check1,pngfilename, size=size, highlightAtoms=ha)
    else:
        Chem.Draw.MolToFile(check1,pngfilename, size=size)
    subprocess.call('imgcat %s' % pngfilename ,shell=True)

    return pngfilename

def Embedfrom2Dto3D(mol):
    #params = AllChem.ETKDGv3()
    #params.useSmallRingTorsions = True
    #AllChem.RemoveStereochemistry(mol)
    #AllChem.AssignAtomChiralTagsFromStructure(mol,confId=-1,replaceExistingTags=True)
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=False, useBasicKnowledge=False)
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s',nonBondedThresh=1000)
    
    return mol

def MMFF94s_energy(mols):
    mols_dict = {}
    mols = [AllChem.AddHs(mol) for mol in mols]
    for i, mol in enumerate(mols):
        AllChem.EmbedMolecule(mol, useRandomCoords=True, useBasicKnowledge=False)
        result = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s',nonBondedThresh=1000)
        mols_dict[mol] = result[0][1]
    
    mols_en = np.array([[mol, en] for mol, en in sorted(mols_dict.items(), key=lambda item: item[1])])
    return mols_en

def Embedfrom2Dto3D_conformers(mol, numConfs=10):
    all_conformers = []
    all_conf_res = []
    mol = AllChem.AddHs(AllChem.RemoveHs(mol))
    cids = AllChem.EmbedMultipleConfs(mol, clearConfs=True, numConfs=numConfs, useRandomCoords=True, useBasicKnowledge=False)
    for i, cid in enumerate(cids):
        mol = AllChem.AddHs(mol)
        res = AllChem.MMFFOptimizeMolecule(mol, confId=cid, mmffVariant='MMFF94s',nonBondedThresh=1000)
        all_conf_res.append(res)
        pdbblock = AllChem.MolToPDBBlock(mol, confId=cid)
        new_mol = AllChem.MolFromPDBBlock(pdbblock, removeHs=False)
        all_conformers.append(new_mol)
    return all_conformers, all_conf_res

def PAH_size(mol):

    grouping = [[0.0,3.0], #pop1
                [3.0,4.5], #pop2
                [4.5,6.0], #pop3
                [6.0,7.5], #pop4
                [7.5,11.5],  #pop5
                [11.5,14.5], #pop6
                [14.5,17.5], #pop7
                [17.5,20.5], #pop8
                [20.5,24.5], #pop9
                ]

    mol = AllChem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol)
    c = mol.GetConformer()
    positions = c.GetPositions()
    xy_pos = positions[:,:2]
    bounding_box = MinimumBoundingBox(xy_pos)
    edge = sorted([bounding_box.length_parallel, bounding_box.length_orthogonal])
    ratio = edge[0] / edge[1]
    group_value = np.mean(edge)
    for j, group_range in enumerate(grouping):
        if group_value >= float(group_range[0]) and group_value < float(group_range[1]):
            return "pop%d" % (j+1), group_value, bounding_box.corner_points




