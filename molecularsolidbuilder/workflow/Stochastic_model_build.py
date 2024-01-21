#Include all functions for coal model build...

import os, sys, glob, subprocess, math, random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from ..Hydrocarbons_Builder import *
from ..Combine_Subunits import Combine_Random, Combine_MD
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