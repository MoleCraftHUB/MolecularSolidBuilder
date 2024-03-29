#Include all functions for coal model build...

import os, sys, glob, subprocess, math, random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from ..Hydrocarbons_Builder import *
from ..Combine_Subunits import Combine_Random, Combine_MD, Get_combine
from ..PDBfile_Converter import PDBImageFileToMols, MolsToPDBImageFile
from ..Heteroatom_Exchanger import *
from ..Examine_CNMR import *
from ..Utility import Embedfrom2Dto3D, Plot_2Dmol

def Expand_PAH_numbers(target_numbers,save_pdb_file=None,save_dir=None):

    acc_mols = []
    for pdb_file, target_number in target_numbers.items():
        pop_loaded = PDBImageFileToMols(pdb_file)
        if target_number != 0:
            pool_num = len(pop_loaded)
            pop_mols_plenty = pop_loaded*target_number
            random.shuffle(pop_mols_plenty)
            pop_mols_new = pop_mols_plenty[:target_number]
            print(f"{len(pop_loaded)} unique molecules in {pdb_file} to {target_number} molecules")
            elements_each = current_element(pop_mols_new)
            nmr_each = carbon_nmr(pop_mols_new, False)
            acc_mols += pop_mols_new
            print(f"Elemental composition (atomic%):{elements_each}")
            print(f"13C-NMR data (ratio):{nmr_each}")
            print("="*50)

    elements_acc = current_element(acc_mols)
    nmr_acc = carbon_nmr(acc_mols, False)
    print(f"From all collected molecules")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        MolsToPDBImageFile(acc_mols,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(acc_mols):
            m = AllChem.AddHs(m,addCoords=True)
            AllChem.MolToPDBFile(m,save_dir+"/mol_%d.pdb" % (j+1))

def Add_faB(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    #mols_modify = deepcopy([AllChem.RemoveHs(mol) for mol in mols_loaded])
    mols_modify = deepcopy(mols_loaded)
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_faB = (t_faB+addtotarget)/t_fapr
    projected_current_faB = c_faB/c_fapr
    print(f"target faB {projected_target_faB:.3%} and current faB {projected_current_faB:.3%}")
    print(f"Add faB carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            mols_list = [mol]
            prob = int(50/(100*abs(projected_target_faB - projected_current_faB)+1))
            if projected_target_faB > projected_current_faB and mw >= 100:
                if random.choice([True]+[False]*prob):
                    mols_list = propagate_new(mol)
                    mols_list = list(sorted(mols_list,key=lambda x:Descriptors.ExactMolWt(x)))
                    random.shuffle(mols_list[:3])
                mols_modify[i] = mols_list[0]
            if projected_target_faB <= projected_current_faB:
                break

        current_nmr = carbon_nmr(mols_modify)
        projected_current_faB = current_nmr['faB'] / current_nmr['fapr']
        print(f"probability to modify: {prob}")
        print(f"iteration {a}: target faB {projected_target_faB:.3%} and current faB {projected_current_faB:.3%}")

        if projected_target_faB <= projected_current_faB:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding faB carbons with {projected_current_faB:.3%}/{projected_target_faB:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [AllChem.AddHs(m,addCoords=True) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))

def Add_faS(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,
            func_smiles=["C(=O)O","CC"],func_percent=[1,0.5],
            ):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy(mols_loaded)
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_faS = (t_faS+addtotarget)/(t_fa)
    projected_current_faS = c_faS/(c_fa)
    print(f"target faS {projected_target_faS:.3%} and current faS {projected_current_faS:.3%}")
    print(f"Add funcs for faS carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            flag = False
            prob = int(50/(100*abs(projected_target_faS - projected_current_faS)+1))
            if projected_target_faS > projected_current_faS and mw >= 0:
                for fs, fr in zip(func_smiles,func_percent):
                    if random.choice([True]+[False]*int(prob*fr)):
                        flag, mol_list = Heteroatom_Aromatic_Func_Add_list(mol,fs,False)
                        random.shuffle(mol_list)
                        mol = mol_list[0]
                        mols_modify[i] = mol_list[0]
                    else:
                        mols_modify[i] = mol
            
            if projected_target_faS <= projected_current_faS:
                break

        current_nmr = carbon_nmr(mols_modify)
        projected_current_faS = (current_nmr['faS'])/(current_nmr['fa'])
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target faS {projected_target_faS:.3%} and current faS {projected_current_faS:.3%}")
        if projected_target_faS <= projected_current_faS:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding faS carbons with {projected_current_faS:.3%}/{projected_target_faS:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))

def Add_falH(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy(mols_loaded)
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_falH = (t_falH+addtotarget) / t_fa
    projected_current_falH = (c_falH) / c_fa
    print(f"target falH {projected_target_falH:.3%} and current falH {projected_current_falH:.3%}")
    print(f"Add falH carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            mols_list = [mol]
            prob = int(50/(100*abs(projected_target_falH - projected_current_falH)+1))
            if projected_target_falH > projected_current_falH and mw >= 100:
                if random.choice([True]+[False]*prob):
                    flag, mol = Heteroatom_Add_6Ring_Aliphatic(mol)
                    mols_list = [mol]
                mols_modify[i] = mols_list[0]
            if projected_target_falH <= projected_current_falH:
                break

        current_nmr = carbon_nmr(mols_modify)
        projected_current_falH = (current_nmr['falH'])/(current_nmr['fa'])
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target falH {projected_target_falH:.3%} and current falH {projected_current_falH:.3%}")

        if projected_target_falH <= projected_current_falH:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding falH carbons with {projected_current_falH:.3%}/{projected_target_falH:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))



def Add_faP(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy([AllChem.RemoveHs(m) for m in mols_loaded])
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_faP = (t_faP+addtotarget)/t_fa
    projected_current_faP = c_faP/c_fa
    print(f"target faP {projected_target_faP:.3%} and current faP {projected_current_faP:.3%}")
    print(f"Add faP carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            prob = int(50/(100*abs(projected_target_faP - projected_current_faP)+1))
            if projected_target_faP > projected_current_faP and mw >= 0:
                if random.choice([True]+[False]*prob):
                    flag, mol_list = Heteroatom_Aromatic_Func_Add_list(mol,'O',False)
                    random.shuffle(mol_list)
                    mols_modify[i] = mol_list[0]
            
            if projected_target_faP <= projected_current_faP:
                break
        current_nmr = carbon_nmr(mols_modify)
        projected_current_faP = current_nmr['faP']/current_nmr['fa']
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target faP {projected_target_faP:.3%} and current faP {projected_current_faP:.3%}")

        if projected_target_faP <= projected_current_faP:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding faP carbons with {projected_current_faP:.3%}/{projected_target_faP:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))


def Add_fal_aliphatic_ring(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,
            func_smiles=["C","CC"],func_percent=[1.0,1.0],
            ):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy(mols_loaded)
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_fal = (t_fal+t_faC)/(t_fa+t_fal)
    projected_current_fal = (c_fal+c_faC)/(c_fa+c_fal)
    print(f"target fal {projected_target_fal:.3%} and current fal {projected_current_fal:.3%}")
    print(f"Add funcs for fal carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            flag = False
            prob = int(50/(100*abs(projected_target_fal - projected_current_fal)+1))
            if projected_target_fal > projected_current_fal and mw >= 0:
                for fs, fr in zip(func_smiles,func_percent):
                    if random.choice([True]+[False]*int(prob*fr)):
                        flag, mol_list = Heteroatom_AliphaticRing_Func_Add_list(mol,fs,False)
                        random.shuffle(mol_list)
                        mol = mol_list[0]
                        mols_modify[i] = mol_list[0]
                    else:
                        mols_modify[i] = mol
            
            if projected_target_fal <= projected_current_fal:
                break

        current_nmr = carbon_nmr(mols_modify)
        projected_current_fal = (current_nmr['fal']+current_nmr['faC'])/(current_nmr['fa']+current_nmr['fal'])
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target fal {projected_target_fal:.3%} and current fal {projected_current_fal:.3%}")
        if projected_target_fal <= projected_current_fal:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding fal carbons with {projected_current_fal:.3%}/{projected_target_fal:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))

def Add_faC1(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy(mols_loaded)
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_faC = (t_faC+addtotarget)/(t_fa + t_fal)
    projected_current_faC = c_faC/(c_fa + c_fal)
    print(f"target faC1 {projected_target_faC:.3%} and current faC1 {projected_current_faC:.3%}")
    print(f"Add faC1 carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            flag = False
            prob = int(50/(100*abs(projected_target_faC - projected_current_faC)+1))
            if projected_target_faC > projected_current_faC and mw >= 0:
                if random.choice([True]+[False]*int(prob)):
                    flag, mol_list = Heteroatom_Func_Add_DoubleO_list(mol,False)
                    random.shuffle(mol_list)
                    mol = mol_list[0]
                    mols_modify[i] = mol_list[0]
                else:
                    mols_modify[i] = mol
            
            if projected_target_faC <= projected_current_faC:
                break

        current_nmr = carbon_nmr(mols_modify)
        projected_current_faC = (current_nmr['faC'])/(current_nmr['fa']+current_nmr['fal'])
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target faC1 {projected_target_faC:.3%} and current faC1 {projected_current_faC:.3%}")
        if projected_target_faC <= projected_current_faC:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding faS carbons with {projected_target_faC:.3%}/{projected_current_faC:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))


def Add_faC2(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy([AllChem.RemoveHs(m) for m in mols_loaded])
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_faC = (t_faC+addtotarget)/(t_fa + t_fal)
    projected_current_faC = c_faC/(c_fa + c_fal)
    print(f"target faC2 {projected_target_faC:.3%} and current faC2 {projected_current_faC:.3%}")
    print(f"Add faC2 carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            flag = False
            prob = int(50/(100*abs(projected_target_faC - projected_current_faC)+1))
            if projected_target_faC > projected_current_faC and mw >= 0:
                if random.choice([True]+[False]*int(prob)):
                    flag, mol_list = Add_OH_tofaC(mol)
                    random.shuffle(mol_list)
                    mols_modify[i] = mol_list[0]
                else:
                    mols_modify[i] = mol
            
            if projected_target_faC <= projected_current_faC:
                break

        current_nmr = carbon_nmr(mols_modify)
        projected_current_faC = (current_nmr['faC'])/(current_nmr['fa']+current_nmr['fal'])
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target faC2 {projected_target_faC:.3%} and current faC2 {projected_current_faC:.3%}")
        if projected_target_faC <= projected_current_faC:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding faS carbons with {projected_target_faC:.3%}/{projected_current_faC:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))


def Add_OHwithfaC(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy([AllChem.RemoveHs(m) for m in mols_loaded])
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_O = target_elements['O']/target_elements['C']
    projected_current_O = current_elements['O']/current_elements['C']
    print(f"target O/C {projected_target_O:.3%} and current O/C {projected_current_O:.3%}")
    print(f"Add O with faC carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            flag = False
            prob = int(50/(100*abs(projected_target_O - projected_current_O)+1))
            if projected_target_O > projected_current_O and mw >= 0:
                if random.choice([True]+[False]*int(prob)):
                    flag, mol_list = Add_OH_tofaC(mol)
                    random.shuffle(mol_list)
                    mols_modify[i] = mol_list[0]
                else:
                    mols_modify[i] = mol
            
            if projected_target_O <= projected_current_O:
                break

        current_nmr = carbon_nmr(mols_modify)
        projected_current_O = current_elements['O']/current_elements['C']
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target faC2 {projected_target_O:.3%} and current faC2 {projected_current_O:.3%}")
        if projected_target_O <= projected_current_O:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding O with faC carbons: {projected_target_O:.3%}/{projected_current_O:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))


def Add_falO(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy(mols_loaded)
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    c_fa     = current_cnmr['fa']
    c_fapr   = current_cnmr['fapr']
    c_faH    = current_cnmr['faH']
    c_faN    = current_cnmr['faN']
    c_faP    = current_cnmr['faP']
    c_faC    = current_cnmr['faC']
    c_faS    = current_cnmr['faS']
    c_faB    = current_cnmr['faB']
    c_fal    = current_cnmr['fal']
    c_falH   = current_cnmr['falH']
    c_falO   = current_cnmr['falO']
    c_falast = current_cnmr['fal*']

    t_fa     = target_cnmr['fa']
    t_fapr   = target_cnmr['fapr']
    t_faP    = target_cnmr['faP']
    t_faC    = target_cnmr['faC']
    t_faS    = target_cnmr['faS']
    t_faB    = target_cnmr['faB'] 
    t_fal    = target_cnmr['fal']
    t_falH   = target_cnmr['falH']
    t_falO   = target_cnmr['falO']
    t_falast = target_cnmr['fal*']

    projected_target_falO = (t_falO+addtotarget) / (t_fa+t_fal)
    projected_current_falO = c_falO / (c_fa+c_fal)
    projected_target_O = target_elements['O']/target_elements['C']
    projected_current_O = current_elements['O']/current_elements['C']
    print(f"target falO {projected_target_falO:.3%} and current falO {projected_current_falO:.3%}")
    print(f"target O/C {projected_target_O:.3%} and current O/C {projected_current_O:.3%}")
    print(f"Add falO carbons to {load_pdb_file}")
    a = 0
    while True:
        a += 1
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            flag = False
            prob = int(50/(100*abs(projected_target_falO - projected_current_falO)+1))
            if projected_target_falO > projected_current_falO and mw >= 0:
                if random.choice([True]+[False]*int(prob)):
                    flag, mol_list = Heteroatom_Aliphatic_Func_Add_withSym_list2(mol,'C','O',2,False)
                    random.shuffle(mol_list)
                    mol = mol_list[0]
                    mols_modify[i] = mol_list[0]
                else:
                    mols_modify[i] = mol
            
            if projected_target_falO <= projected_current_falO:
                break

        current_nmr = carbon_nmr(mols_modify)
        current_elements = current_element(mols_modify)
        projected_current_falO = current_nmr['falO'] / (current_nmr['fa'] + current_nmr['fal'])
        projected_current_O = current_elements['O']/current_elements['C']
        #print(f"probability to modify: {prob}")
        print(f"iteration {a}: target falO {projected_target_falO:.3%} and current falO {projected_current_falO:.3%}")
        print(f"target O/C {projected_target_O:.3%} and current O/C {projected_current_O:.3%}")
        if projected_target_falO <= projected_current_falO:
            break

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    print(f"Completed Adding O for falO carbons with {projected_target_falO:.3%}/{projected_current_falO:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))

def Add_Heteratom_N(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,
                    types={'pyrrolic_n':0.62,'pyridine_n':0.26,'quaternary_n':0.12}):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy([AllChem.RemoveHs(mt) for mt in mols_loaded])
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    total_num_atoms = sum([len(AllChem.AddHs(m).GetAtoms()) for m in mols_modify])
    target_N = target_elements["N"]
    print(f"Total n of atoms:{total_num_atoms}, Target N atoms:{int(total_num_atoms*target_N/100)}")
    
    target_pyrrolic_n = int(total_num_atoms*target_N*types['pyrrolic_n']/100)
    target_pyridine_n = int(total_num_atoms*target_N*types['pyridine_n']/100)
    target_quaternary_n = int(total_num_atoms*target_N*types['quaternary_n']/100)
    furan_site_count = sum([len(num_5ringO(m)) for m in mols_modify])
    print(f"Target Pyrrolic N atoms:{target_pyrrolic_n}, Target Pyridine N atoms:{target_pyridine_n}, Target Quaternary N atoms:{target_quaternary_n}")
    print(f"Available furan sites:{furan_site_count}")
    print(f"Add N atoms to {load_pdb_file}")
    ######################################################################
    current_pyrrolic_n = target_pyrrolic_n
    a = 0
    while True:
        a += 1
        print(f"iteration {a}:Target Pyrrolic N atoms:{target_pyrrolic_n},Current Pyrrolic N atoms:{current_pyrrolic_n}")
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            n_number_inmol = len([atom for atom in mol.GetAtoms() if atom.GetSymbol()=='N'])
            flag = False
            prob = int(50/(100*abs(target_pyrrolic_n - current_pyrrolic_n)+1))
            if (n_number_inmol == 0) and (current_pyrrolic_n > 0) and (random.choice([True] + [False]*prob)):    
                flag, mol = Heteroatom_Sub_5Ring_fromOtoN(mol)
                if flag:
                    current_pyrrolic_n = current_pyrrolic_n - 1
                    mols_modify[i] = mol
        if current_pyrrolic_n == 0:
            break
    current_elements = current_element(mols_modify)
    print(current_elements)
    ######################################################################
    current_pyridine_n = target_pyridine_n
    a = 0
    while True:
        a += 1
        print(f"iteration {a}:Target Pyridine N atoms:{target_pyridine_n},Current Pyridine N atoms:{current_pyridine_n}")
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            n_number_inmol = len([atom for atom in mol.GetAtoms() if atom.GetSymbol()=='N'])
            flag = False
            prob = int(50/(100*abs(target_pyridine_n - current_pyridine_n)+1))
            if (n_number_inmol == 0) and (current_pyridine_n > 0) and (random.choice([True] + [False]*prob)):    
                flag, mol = Heteroatom_Sub_6Ring_fromCtoN(mol)
                if flag:
                    current_pyridine_n = current_pyridine_n - 1
                    mols_modify[i] = mol
        if current_pyridine_n == 0:
            break
    current_elements = current_element(mols_modify)
    print(current_elements)
    ######################################################################
    current_quaternary_n = target_quaternary_n
    a = 0
    while True:
        a += 1
        print(f"iteration {a}:Target Quaternary N atoms:{target_quaternary_n},Current Quaternary N atoms:{current_quaternary_n}")
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            n_number_inmol = len([atom for atom in mol.GetAtoms() if atom.GetSymbol()=='N'])
            flag = False
            prob = int(50/(100*abs(target_quaternary_n - current_quaternary_n)+1))
            if (n_number_inmol == 0) and (current_quaternary_n > 0) and (random.choice([True] + [False]*prob)):    
                flag, mol_list = Heteroatom_Sub_Quaternary_fromCtoN_revise(mol)
                random.shuffle(mol_list)
                if flag:
                    current_quaternary_n = current_quaternary_n - 1
                    mols_modify[i] = mol_list[0]
        if current_quaternary_n == 0:
            break
    current_elements = current_element(mols_modify)
    ######################################################################

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    #print(f"Completed Adding faS carbons with {projected_target_falO:.3%}/{projected_current_falO:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))


def Add_Heteratom_S(target_cnmr,target_elements,load_pdb_file=None,load_dir=None,save_pdb_file=None,save_dir=None,addtotarget=0,
                    types={'thiophene_s':0.41,'aromatic_s':0.26,'aliphatic_s':0.33}):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)
    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    mols_modify = deepcopy([AllChem.RemoveHs(mt) for mt in mols_loaded])
    current_cnmr = carbon_nmr(mols_modify)
    current_elements = current_element(mols_modify)

    total_num_atoms = sum([len(AllChem.AddHs(m).GetAtoms()) for m in mols_modify])
    target_S = target_elements["S"]
    print(f"Total n of atoms:{total_num_atoms}, Target S atoms:{int(total_num_atoms*target_S/100)}")
    
    target_thiophene_s = int(total_num_atoms*target_S*types['thiophene_s']/100)
    target_aromatic_s = int(total_num_atoms*target_S*types['aromatic_s']/100)
    target_aliphatic_s = int(total_num_atoms*target_S*types['aliphatic_s']/100)
    furan_site_count = sum([len(num_5ringO(m)) for m in mols_modify])
    print(f"Target Thiophene S atoms:{target_thiophene_s}, Target Aromatic S atoms:{target_aromatic_s}, Target Aliphatic S atoms:{target_aliphatic_s}")
    print(f"Available furan sites:{furan_site_count}")
    print(f"Add S atoms to {load_pdb_file}")
    ######################################################################
    current_thiophene_s = target_thiophene_s
    a = 0
    while True:
        a += 1
        print(f"iteration {a}:Target Thiophene S atoms:{target_thiophene_s},Current Thiophene S atoms:{current_thiophene_s}")
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            s_number_inmol = len([atom for atom in mol.GetAtoms() if atom.GetSymbol()=='S'])
            flag = False
            prob = int(50/(100*abs(target_thiophene_s - current_thiophene_s)+1))
            if (s_number_inmol == 0) and (current_thiophene_s > 0) and (random.choice([True] + [False]*prob)):    
                flag, mol = Heteroatom_Sub_5Ring_fromOtoS(mol)
                if flag:
                    current_thiophene_s = current_thiophene_s - 1
                    mols_modify[i] = mol
        if current_thiophene_s == 0:
            break
    current_elements = current_element(mols_modify)
    print(current_elements)
    ######################################################################

    current_aromatic_s = target_aromatic_s
    a = 0
    while True:
        a += 1
        print(f"iteration {a}:Target Aromatic S atoms:{target_aromatic_s},Current Aromatic S atoms:{current_aromatic_s}")
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            s_number_inmol = len([atom for atom in mol.GetAtoms() if atom.GetSymbol()=='S'])
            flag = False
            prob = int(50/(100*abs(target_aromatic_s - current_aromatic_s)+1))
            if (s_number_inmol == 0) and (current_aromatic_s > 0) and (random.choice([True] + [False]*prob)):    
                flag, mol = Heteroatom_Func_SubOHtoSH(mol)
                if flag:
                    current_aromatic_s = current_aromatic_s - 1
                    mols_modify[i] = mol
        if current_aromatic_s == 0:
            break
    current_elements = current_element(mols_modify)
    print(current_elements)
    ######################################################################
    current_aliphatic_s = target_aliphatic_s
    a = 0
    while True:
        a += 1
        print(f"iteration {a}:Target Aliphatic S atoms:{target_aliphatic_s},Current Aliphatic S atoms:{current_aliphatic_s}")
        for i, mol in enumerate(mols_modify):
            mw = Descriptors.MolWt(mol)
            ringinfo = mol.GetRingInfo()
            aring = ringinfo.AtomRings()
            s_number_inmol = len([atom for atom in mol.GetAtoms() if atom.GetSymbol()=='S'])
            flag = False
            prob = int(50/(100*abs(target_aliphatic_s - current_aliphatic_s)+1))
            if (s_number_inmol == 0) and (current_aliphatic_s > 0) and (random.choice([True] + [False]*prob)):    
                flag, mol_list = Heteroatom_Sub_Quaternary_fromCtoN_revise(mol)
                random.shuffle(mol_list)
                if flag:
                    current_aliphatic_s = current_aliphatic_s - 1
                    mols_modify[i] = mol_list[0]
        if current_aliphatic_s == 0:
            break
    current_elements = current_element(mols_modify)
    ######################################################################

    elements_acc = current_element(mols_modify)
    nmr_acc = carbon_nmr(mols_modify, False)
    #print(f"Completed Adding faS carbons with {projected_target_falO:.3%}/{projected_current_falO:.3%}")
    print(f"Elemental composition (atomic%):{elements_acc}")
    print(f"13C-NMR data (ratio):{nmr_acc}")
    print("="*50)

    if save_pdb_file != None:
        print(f"Write as a single PDB file to {save_pdb_file}")
        mols_modify = [Embedfrom2Dto3D(m) for m in mols_modify]
        MolsToPDBImageFile(mols_modify,save_pdb_file)

    if save_dir != None:
        print(f"Write as individual PDB files to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j, m in enumerate(mols_modify):
            AllChem.MolToPDBFile(Embedfrom2Dto3D(m),save_dir+"/mol_%d.pdb" % (j+1))

def Mass_weight_Distribute(target_mass_range, target_mass_dist,load_pdb_file=None,load_dir=None,save_dir='mass_distribute',restart=True):

    if load_pdb_file != None:
        mols_loaded = PDBImageFileToMols(load_pdb_file)

    if load_dir != None:
        pdbfile_loaded = sorted(glob.glob(load_dir+'/*.pdb'),key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)),reverse=True)
        mols_loaded = [AllChem.MolFromPDBFile(pdbfile) for pdbfile in pdbfile_loaded]

    dirs = []
    empty_masspermol = 0

    if (not os.path.isdir(save_dir)) and (not restart):
        os.makedirs('./%s/seed_molecules' % save_dir)
        for mass_dir in target_mass_range.tolist():
            os.makedirs('./%s/%s' % (save_dir,mass_dir))
            dirs.append('./%s/%s' % (save_dir,mass_dir))
    elif (os.path.isdir(save_dir)) and (not restart):
        raise FileExistsError(f"The directory '{save_dir}' already exists.")
    elif restart:
        print('restart')
        for mass_dir in target_mass_range.tolist():
            dirs.append('./%s/%s' % (save_dir,mass_dir))
    working_dir = './%s' % save_dir
    
    if not restart:
        for i, mol in enumerate(mols_loaded):
            mol_H = AllChem.AddHs(mol, addCoords=True)
            AllChem.MolToPDBFile(mol_H,'./%s/seed_molecules/mol_%d.pdb' % (save_dir,i+1))
        target_total_mass = [np.mean([int(t) for t in m.split('-')])*c for m, c in zip(target_mass_range.tolist(), target_mass_dist.tolist())]

    mass_x = target_mass_range.tolist()
    target_count = target_mass_dist.tolist()
    current_count = [ len([pdb for pdb in glob.glob(dir_+'/combined_mol_*.pdb')]) for dir_ in dirs ]
    df_dict = {"mass":mass_x,"target":target_count,"current":current_count}
    df = pd.DataFrame.from_dict(df_dict).transpose()

    pdb_files_fresh = sorted([pdb for pdb in glob.glob('%s/seed_molecules/mol_*.pdb' % working_dir)],key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)))
    check_weight_fresh = [Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in pdb_files_fresh]

    while (len(pdb_files_fresh) > 0):
        for i, mass in enumerate(mass_x):
            mass_m = np.mean([int(mass.split('-')[0]),int(mass.split('-')[1])])
            mass_l = int(mass.split('-')[0])
            mass_h = int(mass.split('-')[1])
            pdb_files_fresh = sorted([pdb for pdb in glob.glob('%s/seed_molecules/mol_*.pdb' % working_dir)],key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)))
            check_weight_fresh = [Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in pdb_files_fresh]
            pdb_files_available = [pdb_f for pdb_f in pdb_files_fresh if Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) < mass_h]
            check_weight_available = [Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in pdb_files_available]

            mass_repo = sum(check_weight_fresh)

            if mass_repo > mass_h and len(check_weight_available) > 0 and len(check_weight_fresh) > 0:
                print(mass,target_count[i],current_count[i],len(check_weight_available))
                target_c = target_count[i]
                dir_ = working_dir+'/'+mass
                current_c = len([pdb for pdb in glob.glob(dir_+'/combined_mol_*.pdb')])
                tt = target_c - current_c
                for j in range(tt):
                    collected_pdbs = []
                    collected_mass = 0
                    current_c = len([pdb for pdb in glob.glob(dir_+'/combined_mol_*.pdb')])

                    lmw = 0
                    trial = 0
                    while True:
                        random.shuffle(pdb_files_available)
                        if len(pdb_files_available) > 0:
                            for k in range(100):
                                subset = pdb_files_available[:1]
                                subset_mw = [Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in subset]+[empty_masspermol]*(len(subset)-1)
                                mw_check = sum(subset_mw)
                                if mw_check < (mass_h - collected_mass) and mw_check > 0:
                                    [collected_pdbs.append(pdb) for pdb in subset]
                                    [pdb_files_available.remove(pdb) for pdb in subset]
                                    collected_mass = sum([Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in collected_pdbs]\
                                            +[empty_masspermol]*(len(collected_pdbs)-1))
                                    if (mass_h > collected_mass) and (mass_l+10 <=collected_mass) and len(collected_pdbs) > 0:
                                        break
                        trial += 1

                        if trial > 20:
                            collected_pdbs = []
                            collected_mass = 0
                            pdb_files_fresh = sorted([pdb for pdb in glob.glob('%s/seed_molecules/mol_*.pdb' % working_dir)],key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)))
                            pdb_files_available = [pdb_f for pdb_f in pdb_files_fresh if Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) < mass_h]
                            mw_split1 = [Descriptors.MolWt(AllChem.MolFromPDBFile("./"+pdb_f)) for pdb_f in pdb_files_available]
                            random.shuffle(pdb_files_available)

                            trial = 0
                            lmw = 0
                        if (mass_h > collected_mass) and (mass_l+10 <=collected_mass) and len(collected_pdbs) > 0:
                            break

                    collected_mass = sum([Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in collected_pdbs]\
                            +[empty_masspermol]*(len(collected_pdbs)-1))
                    
                    print('collected',mass_l, collected_mass, collected_pdbs)

                    cmd2 = "rm "+" ".join(collected_pdbs)
                    cmd1 = "cat "+" ".join(collected_pdbs) + " > %s/combined_mol_%d.pdb" % (dir_,current_c)
                    subprocess.call(cmd1,shell=True)
                    subprocess.call(cmd2,shell=True)

                    pdb_files_fresh = sorted([pdb for pdb in glob.glob('%s/seed_molecules/mol_*.pdb' % working_dir)],key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)))
                    pdb_files_available = [pdb_f for pdb_f in pdb_files_fresh if Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) < mass_h]
                    current_c = len([pdb for pdb in glob.glob(dir_+'/combined_mol_*.pdb')])
                    print(mass, target_c, current_c)

            elif mass_repo < mass_h and mass_repo > 0:
                if mass_repo < mass_l+10:
                    continue
                elif mass_repo >= mass_l+10:
                    collected_pdbs = pdb_files_fresh
                    cmd1 = "cat "+" ".join(collected_pdbs) + " > %s/combined_mol_%d.pdb" % (dir_,current_c)
                    cmd2 = "rm "+" ".join(collected_pdbs)
                    subprocess.call(cmd1,shell=True)
                    subprocess.call(cmd2,shell=True)

            elif len(check_weight_available) == 0:
                continue
        
        pdb_files_fresh = sorted([pdb for pdb in glob.glob('%s/seed_molecules/mol_*.pdb' % working_dir)],key=lambda x:Descriptors.MolWt(AllChem.MolFromPDBFile(x)))
        check_weight_fresh = [Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in pdb_files_fresh]
        pdb_files_available = [pdb_f for pdb_f in pdb_files_fresh if Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) < mass_h]
        check_weight_available = [Descriptors.MolWt(AllChem.MolFromPDBFile(pdb_f)) for pdb_f in pdb_files_available]

        current_count = [ len([pdb for pdb in glob.glob(dir_+'/combined_mol_*.pdb')]) for dir_ in dirs ]
        flag = [True if tc-cc <= 0 else False for tc, cc in zip(target_count, current_count)]
        if all(flag):
            for i in range(len(target_count)):
                if target_count[i] != 0:
                    target_count[i] += 1
            for i in range(len(target_count),0,-1):
                if target_count[i-1] > 0 and target_count[i] == 0:
                    target_count[i] += 1

        flag = [True if tc-cc <= 0 else False for tc, cc in zip(target_count, current_count)]

def Combine_Mols(load_dir='mass_distribute',save_dir='mass_distribute_combine',method='smi'):
       
    seperated_mols = sorted(glob.glob(load_dir+'/*-*/*.pdb'),key=lambda x:int(x.split('/')[-2].split('-')[0]))
    for seperate_mol in seperated_mols:
        combined_mol_path = save_dir+'/'+seperate_mol.split('/')[-2]
        if not os.path.exists(combined_mol_path):
            os.makedirs(combined_mol_path)
        k = int(seperate_mol[:-4].split('_')[-1])
        #print(combined_mol_path,k)
        result = Get_combine(pdb_file=seperate_mol,path=combined_mol_path,link_sms=[''], base_num=k, estimate=method)
        print(result)

