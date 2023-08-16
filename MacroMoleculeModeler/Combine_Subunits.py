import os, sys, glob, subprocess, random
from tkinter import simpledialog
import numpy as np
from itertools import combinations
from copy import deepcopy
from ase.io import read, write
from ase import Atoms, Atom
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

from .Hydrocarbons_Builder import convex_bond_atom
from .Run_MD import ReaxFFminimize 
from .Utility import Plot_2Dmol, Embedfrom2Dto3D_conformers, Embedfrom2Dto3D, MMFF94s_energy
from .Heteroatom_Exchanger import *

def molwith_idx(mol):
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(atom.GetIdx())
	return mol

def MMFFs_3Dconstruct(mol):

    #lmp_path = 'lmp_path'
    mol2 = AllChem.AddHs(mol)
    #run ETKDG 100 times
    cids = AllChem.EmbedMultipleConfs(mol2,numConfs=10,numThreads=0,useRandomCoords=True)
    rmslist = []
    AllChem.AlignMolConformers(mol2,RMSlist=rmslist)
    test = AllChem.MMFFOptimizeMoleculeConfs(mol2,numThreads=0)
    return mol2

def Get_combine(pdb_file, path, link_sms, base_num=0, estimate='energy'):

	f = open(pdb_file,'r')
	lines = f.readlines()
	end_index = [l for l in range(len(lines)) if 'END' in lines[l]]
	start = 0

	ms = []
	for i in range(len(end_index)):
		end = end_index[i] + 1
		pdb_block = lines[start:end]
		start = end
		pdb_block_str = "".join(pdb_block)
		m = AllChem.MolFromPDBBlock(pdb_block_str)
		m = AllChem.AddHs(m)
		ms.append(m)

	for i in range(1):
		if estimate =='energy':
			mol = Combine_MMFF94s(ms)
		else:
			mol = Combine_SMILES(ms)

		if False:
			mols = Embedfrom2Dto3D_conformers(mol)
			pdb_block = "".join([AllChem.MolToPDBBlock(mol) for mol in mols])

		mol = Embedfrom2Dto3D(mol)
		pdb_block = AllChem.MolToPDBBlock(mol)
		pdb_file = open(path+'/molecule_%d.pdb' % base_num,'w')
		pdb_file.write(pdb_block)
		pdb_file.close()

	return path+'/molecule_%d.pdb' % base_num

def Sidechain_idx_sym(mol1):

	mol1 = AllChem.RemoveHs(mol1)
	atoms1 = mol1.GetAtoms()
	avail_atom_idx_1 = [atom.GetIdx() for atom in atoms1 \
		if atom.GetTotalNumHs() >= 1 and not atom.IsInRing() \
			and not atom.GetIsAromatic() \
			and len([n for n in atom.GetNeighbors()]) == 1]

	avail_atom_idx_2 = [atom.GetIdx() for atom in atoms1 \
		if atom.GetTotalNumHs() >= 1 \
			and atom.IsInRing() \
			and len([n for n in atom.GetNeighbors()]) < 3 ]

	if len(avail_atom_idx_1) == 0:
		avail_atom_idx_3 = avail_atom_idx_2
	else:
		avail_atom_idx_3 = avail_atom_idx_1

	#avail_atom_idx_3 = avail_atom_idx_1 + avail_atom_idx_2
	avail_atom_sym_3 = [atoms1[idx].GetSymbol() for idx in avail_atom_idx_3]
	avail_atoms_dict1 = {idx:sym for idx, sym in zip(avail_atom_idx_3, avail_atom_sym_3)}
	
	return avail_atoms_dict1

def type_idx_sym(mol1):
	mol1 = AllChem.RemoveHs(mol1)
	atoms1 = mol1.GetAtoms()
	avail_atom_idx_chain = [atom.GetIdx() for atom in atoms1 \
		if atom.GetTotalNumHs() >= 1 and not atom.IsInRing() \
			and not atom.GetIsAromatic() \
			and len([n for n in atom.GetNeighbors()]) == 1]
	avail_atom_sym_chain = [atoms1[idx].GetSymbol() for idx in avail_atom_idx_chain]
	avail_atoms_dict_chain = {idx:sym for idx, sym in zip(avail_atom_idx_chain, avail_atom_sym_chain)}

	avail_atom_idx_ring = [atom.GetIdx() for atom in atoms1 \
		if atom.GetTotalNumHs() >= 1 \
			and atom.IsInRing() \
			and len([n for n in atom.GetNeighbors()]) < 3 ]
	avail_atom_sym_ring = [atoms1[idx].GetSymbol() for idx in avail_atom_idx_ring]
	avail_atoms_dict_ring = {idx:sym for idx, sym in zip(avail_atom_idx_ring, avail_atom_sym_ring)}

	return avail_atoms_dict_chain, avail_atoms_dict_ring


def Combine_MMFF94s(ms):
	#ms is the list of mol objects
	#shuffle the molecules before combine
	if len(ms) == 1:
		#smi = AllChem.MolToSmiles(AllChem.RemoveHs(ms[0]))
		new_mol = AllChem.AddHs(ms[0])
		return new_mol

	elif len(ms) > 1:
		np.random.shuffle(ms)
		seed = ms[0]
		seed = AllChem.RemoveHs(seed)

		for i in range(1,len(ms)):
			ms[i] = AllChem.RemoveHs(ms[i])
			new_mol = ms[i]
			mols_en = Crosslink_MMFF94s(seed,new_mol)
			new_seed = mols_en[0][0]
			seed = deepcopy(new_seed)
		new_mol = AllChem.AddHs(seed)

		return new_mol

def Crosslink_MMFF94s(mol1,mol2):
	
	combined_mol = []
	mol1 = AllChem.RemoveHs(mol1)
	atoms = mol1.GetAtoms()
	a1c, a1r = type_idx_sym(mol1)

	mol2 = AllChem.RemoveHs(mol2)
	a2c, a2r = type_idx_sym(mol2)

	pair_idx = []
	pair_symbol = []
	exclude = ['SS','SO','OS','NN','NO','ON','OO','SN','NS']
	for a1_ in a1c.items():
		for a2_ in a2c.items():
			if a1_[1]+a2_[1] not in exclude:
				pair_idx.append([a1_[0],a2_[0]])
				pair_symbol.append(a1_[1]+a2_[1])

	if len(pair_idx) == 0:
		for a1_ in a1c.items():
			for a2_ in a2r.items():
				if a1_[1]+a2_[1] not in exclude:
					pair_idx.append([a1_[0],a2_[0]])
					pair_symbol.append(a1_[1]+a2_[1])
		for a1_ in a1r.items():
			for a2_ in a2c.items():
				if a1_[1]+a2_[1] not in exclude:
					pair_idx.append([a1_[0],a2_[0]])
					pair_symbol.append(a1_[1]+a2_[1])
	if len(pair_idx) == 0:
		for a1_ in a1r.items():
			for a2_ in a2r.items():
				if a1_[1]+a2_[1] not in exclude:
					pair_idx.append([a1_[0],a2_[0]])
					pair_symbol.append(a1_[1]+a2_[1])

	pair_new = []
	for pi, ps in zip(pair_idx, pair_symbol):
		pair_new.append(pi)

	pair_update = [[p[0],p[1]+len(atoms)] for p in pair_new] # update index for combined mol
	m_comb = Chem.CombineMols(mol1,mol2)
	for i, link in enumerate(pair_update):
		m_comb_dup = deepcopy(m_comb)
		a1 = link[0]
		a2 = link[1]
		edcombo = Chem.EditableMol(m_comb_dup)
		edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.SINGLE)
		connected_m = edcombo.GetMol()
		atoms_m = connected_m.GetAtoms()
		#if atoms_m[a1].GetSymbol() == 'O' and atoms_m[a2].GetSymbol() == 'O':
		#	atoms_m[a1].SetAtomicNum(6)
		#	atoms_m[a1].SetNumExplicitHs(3)

		connected_m = AllChem.AddHs(connected_m)
		atoms_m = connected_m.GetAtoms()
		hs_remove = []
		#print(atoms_m[a1].GetSymbol(),atoms_m[a2].GetSymbol())
		#print([n.GetSymbol() for n in atoms_m[a1].GetNeighbors()],[n.GetSymbol() for n in atoms_m[a2].GetNeighbors()])
		hs_a1 = max([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
		hs_a2 = max([n.GetIdx() for n in atoms_m[a2].GetNeighbors() if n.GetSymbol() == 'H'])
		hs_remove.append(hs_a1)
		hs_remove.append(hs_a2)

		hs_remove = list(sorted(set(hs_remove), reverse=True))
		edcombo2 = Chem.EditableMol(connected_m)
		[ edcombo2.RemoveAtom(h_idx) for h_idx in hs_remove ]
		connected_m = edcombo2.GetMol()
		final = deepcopy(connected_m)
		MW = Descriptors.ExactMolWt(final)
		combined_mol.append(final)

	mols = MMFF94s_energy(combined_mol)
	return mols

def Combine_SMILES(ms):

	#ms is the list of mol objects
	#shuffle the molecules before combine
	if len(ms) == 1:
		#smi = AllChem.MolToSmiles(AllChem.RemoveHs(ms[0]))
		new_mol = AllChem.AddHs(ms[0])
		return new_mol

	elif len(ms) > 1:
		np.random.shuffle(ms)
		seed = ms[0]
		seed = AllChem.RemoveHs(seed)

		for i in range(1,len(ms)):
			ms[i] = AllChem.RemoveHs(ms[i])
			new_mol = ms[i]
			mols = Crosslink_SMILES(seed,new_mol)
			new_seed = random.choice(mols)[0]
			seed = deepcopy(new_seed)
		new_mol = AllChem.AddHs(seed)

		return new_mol

def Crosslink_SMILES(mol1,mol2):
	
	combined_mol = []
	mol1 = AllChem.RemoveHs(mol1)
	atoms = mol1.GetAtoms()
	a1c, a1r = type_idx_sym(mol1)

	mol2 = AllChem.RemoveHs(mol2)
	a2c, a2r = type_idx_sym(mol2)

	pair_idx = []
	pair_symbol = []
	exclude = ['SS','SO','OS','NN','NO','ON','OO','SN','NS']
	for a1_ in a1c.items():
		for a2_ in a2c.items():
			if a1_[1]+a2_[1] not in exclude:
				pair_idx.append([a1_[0],a2_[0]])
				pair_symbol.append(a1_[1]+a2_[1])

	if len(pair_idx) == 0:
		for a1_ in a1c.items():
			for a2_ in a2r.items():
				if a1_[1]+a2_[1] not in exclude:
					pair_idx.append([a1_[0],a2_[0]])
					pair_symbol.append(a1_[1]+a2_[1])
		for a1_ in a1r.items():
			for a2_ in a2c.items():
				if a1_[1]+a2_[1] not in exclude:
					pair_idx.append([a1_[0],a2_[0]])
					pair_symbol.append(a1_[1]+a2_[1])
	if len(pair_idx) == 0:
		for a1_ in a1r.items():
			for a2_ in a2r.items():
				if a1_[1]+a2_[1] not in exclude:
					pair_idx.append([a1_[0],a2_[0]])
					pair_symbol.append(a1_[1]+a2_[1])

	pair_new = []
	for pi, ps in zip(pair_idx, pair_symbol):
		pair_new.append(pi)

	pair_update = [[p[0],p[1]+len(atoms)] for p in pair_new] # update index for combined mol
	m_comb = Chem.CombineMols(mol1,mol2)
	for i, link in enumerate(pair_update):
		m_comb_dup = deepcopy(m_comb)
		a1 = link[0]
		a2 = link[1]
		edcombo = Chem.EditableMol(m_comb_dup)
		edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.SINGLE)
		connected_m = edcombo.GetMol()
		atoms_m = connected_m.GetAtoms()
		#if atoms_m[a1].GetSymbol() == 'O' and atoms_m[a2].GetSymbol() == 'O':
		#	atoms_m[a1].SetAtomicNum(6)
		#	atoms_m[a1].SetNumExplicitHs(3)
		
		connected_m = AllChem.AddHs(connected_m)
		atoms_m = connected_m.GetAtoms()
		hs_remove = []
		#print(atoms_m[a1].GetSymbol(),atoms_m[a2].GetSymbol())
		#print([n.GetSymbol() for n in atoms_m[a1].GetNeighbors()],[n.GetSymbol() for n in atoms_m[a2].GetNeighbors()])
		hs_a1 = max([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
		hs_a2 = max([n.GetIdx() for n in atoms_m[a2].GetNeighbors() if n.GetSymbol() == 'H'])
		hs_remove.append(hs_a1)
		hs_remove.append(hs_a2)

		hs_remove = list(sorted(set(hs_remove), reverse=True))
		edcombo2 = Chem.EditableMol(connected_m)
		[ edcombo2.RemoveAtom(h_idx) for h_idx in hs_remove ]
		connected_m = edcombo2.GetMol()
		final = deepcopy(connected_m)
		MW = Descriptors.ExactMolWt(final)
		combined_mol.append(final)

	mols = [[mol_,0] for mol_ in combined_mol]
	return mols


def Combine_Random(ms, linker_sm=['C']):

	#ms is the list of mol objects
	#shuffle the molecules before combine
	if len(ms) == 1:
		#smi = AllChem.MolToSmiles(AllChem.RemoveHs(ms[0]))
		smi = None
		new_mol = AllChem.AddHs(ms[0])
		return new_mol, smi

	elif len(ms) > 1:
		np.random.shuffle(ms)

		add_crosslink = np.zeros(len(ms))
		seed = ms[0]
		seed = AllChem.RemoveHs(seed)
		index_check = []
		count = np.zeros(len(ms))
		atoms = seed.GetAtoms()
		
		avail_atom_idx_0 = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() == 1 and len([n for n in atom.GetNeighbors()]) < 3 ]
		index_check.append(avail_atom_idx_0)

		for i in range(1,len(ms)):
			linker_sm_iter = random.choice(linker_sm)
			seed = AllChem.RemoveHs(seed)
			atoms = seed.GetAtoms()
			ringinfo = seed.GetRingInfo()
			aring1 = [tmp for tmp in ringinfo.AtomRings() if len(tmp) <= 6]
			avail_atom_idx_1 = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() >= 1 and len([n for n in atom.GetNeighbors()]) < 3 ]

			ms[i] = AllChem.RemoveHs(ms[i])
			atoms2 = ms[i].GetAtoms()
			ringinfo = ms[i].GetRingInfo()
			aring2 = [tmp for tmp in ringinfo.AtomRings() if len(tmp) <= 6]
			avail_atom_idx_2 = [atom2.GetIdx() for atom2 in atoms2 if atom2.GetTotalNumHs() >= 1 and len([n for n in atom2.GetNeighbors()]) < 3 ]
			index_check.append(list(avail_atom_idx_2))

			#test1 = [idx for idx in avail_atom_idx_1 if not atoms[idx].GetIsAromatic() and len([r for r in aring1 if idx in r]) == 0 ]
			#test2 = [idx for idx in avail_atom_idx_2 if not atoms2[idx].GetIsAromatic() and len([r for r in aring1 if idx in r]) == 0 ] 

			linker_sm_tmp = ''
			linker_sm_new = linker_sm_iter

			while True:
				linker_sm_new = linker_sm_iter

				avail_atom_idx_1_screen = [idd for idd in avail_atom_idx_1 \
					if not (atoms[idd].GetTotalNumHs() == 2 and len([r for r in aring1 if atoms[idd].GetIdx() in r]) <= 1)]
				avail_atom_idx_1_screen = [idd for idd in avail_atom_idx_1_screen \
					if not (atoms[idd].GetTotalNumHs() == 1 and atoms[idd].GetSymbol()=='C' and len([r for r in aring1 if atoms[idd].GetIdx() in r]) == 0)]
				avail_atom_idx_1_screen = [idd for idd in avail_atom_idx_1_screen if atoms[idd].GetSymbol()!='N']

				avail_atom_idx_2_screen = [idd for idd in avail_atom_idx_2 \
					if not (atoms2[idd].GetTotalNumHs() == 2 and len([r for r in aring2 if atoms2[idd].GetIdx() in r]) <= 1)]
				avail_atom_idx_2_screen = [idd for idd in avail_atom_idx_2_screen \
					if not (atoms2[idd].GetTotalNumHs() == 1 and atoms2[idd].GetSymbol()=='C' and len([r for r in aring2 if atoms2[idd].GetIdx() in r]) == 0)]
				avail_atom_idx_2_screen = [idd for idd in avail_atom_idx_2_screen if atoms2[idd].GetSymbol()!='N']

				a1_random = list(sorted([idd for idd in avail_atom_idx_1_screen],key=lambda x: atoms[x].GetTotalNumHs(),reverse=True))[0]
				a2_random = list(sorted([idd for idd in avail_atom_idx_2_screen],key=lambda x: atoms2[x].GetTotalNumHs(),reverse=True))[0]

				if (atoms[a1_random].GetTotalNumHs() >= 1 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() >= 1 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 1 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 1 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 3 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() == 3 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 1 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() == 3 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 1 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 3 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() == 1 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 1 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetSymbol() =='O' and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 1 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetSymbol() =='C' and atoms2[a2_random].GetSymbol() =='O') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 1 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 1 and atoms[a1_random].GetSymbol() =='O') \
					and (atoms2[a2_random].GetTotalNumHs() == 3 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 3 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() == 1 and atoms2[a2_random].GetSymbol() =='O') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				#Sulfur
				elif (atoms[a1_random].GetTotalNumHs() == 1 and atoms[a1_random].GetSymbol() =='S') \
					and (atoms2[a2_random].GetTotalNumHs() == 1 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 1 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 1 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() == 1 and atoms2[a2_random].GetSymbol() =='S') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 1 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 1 and atoms[a1_random].GetSymbol() =='S') \
					and (atoms2[a2_random].GetTotalNumHs() == 3 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 3 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() == 1 and atoms2[a2_random].GetSymbol() =='S') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 1 and atoms[a1_random].GetSymbol() =='S') \
					and (atoms2[a2_random].GetTotalNumHs() == 3 and atoms2[a2_random].GetSymbol() =='C') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetTotalNumHs() == 3 and atoms[a1_random].GetSymbol() =='C') \
					and (atoms2[a2_random].GetTotalNumHs() == 1 and atoms2[a2_random].GetSymbol() =='S') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				elif (atoms[a1_random].GetSymbol() =='O' and atoms2[a2_random].GetSymbol() =='O') \
					and len([r for r in aring1 if atoms[a1_random].GetIdx() in r]) == 0 \
					and len([r for r in aring2 if atoms2[a2_random].GetIdx() in r]) == 0 :
					linker_sm_tmp = ''
					break
				else:
					print('p',atoms[a1_random].GetSymbol(), atoms2[a2_random].GetSymbol(), linker_sm_new)
					print('p',atoms[a1_random].GetTotalNumHs(), atoms2[a2_random].GetTotalNumHs())
					continue

			linker_sm_new = linker_sm_tmp
			#print(atoms[a1_random].GetSymbol(), atoms2[a2_random].GetSymbol(), linker_sm_new)
			#print(atoms[a1_random].GetTotalNumHs(), atoms2[a2_random].GetTotalNumHs())
			#Plot_2Dmol(molwith_idx(seed))
			#Plot_2Dmol(molwith_idx(ms[i]))

			m_comb = Chem.CombineMols(seed,ms[i])
			a2_random = a2_random + len(atoms) # update index for combined mol

			if linker_sm_new != '':
				linker = AllChem.MolFromSmiles(linker_sm_new)
				linker_indx = np.array([atom.GetIdx() for atom in linker.GetAtoms()]) + len(m_comb.GetAtoms())
				linker_indx = [int(l) for l in linker_indx]
				m_comb2 = Chem.CombineMols(m_comb,linker)
			elif linker_sm_new == '':
				linker_indx = []
				m_comb2 = m_comb

			atoms_comb2 = m_comb2.GetAtoms()
			
			#Plot_2Dmol(m_comb2)
			edcombo = Chem.EditableMol(m_comb2)
			a1 = int(a1_random)
			a2 = int(a2_random)
			b1 = None 
			b2 = None

			if linker_sm_new != '':
				b1 = int(linker_indx[0])
				b2 = int(linker_indx[-1])
				edcombo.AddBond(a1,b1,order=Chem.rdchem.BondType.SINGLE)
				edcombo.AddBond(a2,b2,order=Chem.rdchem.BondType.SINGLE)
			else:
				edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.SINGLE)
				
			connected_m = edcombo.GetMol()
			atoms_m = connected_m.GetAtoms()
			if atoms_m[a1].GetSymbol() == 'O' and atoms_m[a2].GetSymbol() == 'O':
				atoms_m[a1].SetAtomicNum(6)
				atoms_m[a1].SetNumExplicitHs(3)
				atoms_m[a2].SetAtomicNum(6)
				atoms_m[a2].SetNumExplicitHs(3)

			connected_m = AllChem.AddHs(connected_m)
			atoms_m = connected_m.GetAtoms()
			hs_remove = []
			hs_a1 = max([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
			hs_a2 = max([n.GetIdx() for n in atoms_m[a2].GetNeighbors() if n.GetSymbol() == 'H'])
			hs_remove.append(hs_a1)
			hs_remove.append(hs_a2)

			if b1 != None and b2 != None and b1 == b2:
				hs_b1 = [n.GetIdx() for n in atoms_m[b1].GetNeighbors() if n.GetSymbol() == 'H']
				hs_b2 = [n.GetIdx() for n in atoms_m[b2].GetNeighbors() if n.GetSymbol() == 'H']
				for tm in hs_b1[:2]:
					hs_remove.append(tm)

			if b1 != None and b2 != None and b1 != b2:
				hs_b1 = max([n.GetIdx() for n in atoms_m[b1].GetNeighbors() if n.GetSymbol() == 'H'])
				hs_b2 = max([n.GetIdx() for n in atoms_m[b2].GetNeighbors() if n.GetSymbol() == 'H'])
				hs_remove.append(hs_b1)
				hs_remove.append(hs_b2)

			hs_remove = list(sorted(set(hs_remove), reverse=True))
			edcombo2 = Chem.EditableMol(connected_m)
			[ edcombo2.RemoveAtom(h_idx) for h_idx in hs_remove ]
			connected_m = edcombo2.GetMol()
			#Plot_2Dmol(connected_m)

			seed = deepcopy(connected_m)
			for j in range(len(index_check)):
				ms_inds = index_check[j]
				if a1 in ms_inds:
					count[j] += 1
			MW = Descriptors.ExactMolWt(seed)
			#print(MW)

		#inchi = AllChem.MolToInchi(connected_m)
		#smi = AllChem.MolToSmiles(AllChem.RemoveHs(connected_m))
		smi = None
		#final_mol_3d = MMFFs_3Dconstruct(connected_m)[0]
		#s = AllChem.MolToPDBBlock(final_mol_3d)
		return connected_m, smi

	#Count the number of bonds added to each molecule


def Combine_MD(ms, linker_sm='CCOCC', label=0):

	#ms is the list of mol objects
	#shuffle the molecules before combine
	#np.random.shuffle(ms)

	add_crosslink = np.zeros(len(ms))
	seed = ms[0]
	index_check = []
	count = np.zeros(len(ms))
	convex_bond, convex_atom = convex_bond_atom(seed)
	atoms = seed.GetAtoms()
	avail_atom_idx_0 = [idx for idx in convex_atom if atoms[idx].IsInRing()]
	index_check.append(avail_atom_idx_0)

	for i in range(1,len(ms)):
		seed = AllChem.RemoveHs(seed)
		convex_bond, convex_atom = convex_bond_atom(seed)
		atoms = seed.GetAtoms()
		#avail_atom_idx_1 = [idx for idx in convex_atom if atoms[idx].IsInRingSize(6)]
		avail_atom_idx_1 = [idx for idx in convex_atom if atoms[idx].IsInRingSize(6) or (atoms[idx].IsInRingSize(5) and atoms[idx].GetTotalNumHs() == 1)]
		#check1 = deepcopy(seed)
		#check1 = AllChem.RemoveHs(check1)
		#AllChem.Compute2DCoords(check1)
		#Chem.rdCoordGen.AddCoords(check1)
		#Chem.Draw.MolToFile(check1,'./%d_before.png' % i, size=(800,800),kekulize=True,highlightAtoms=avail_atom_idx_1)#, highlightBonds=convex_bond)

		atoms2 = ms[i].GetAtoms()
		convex_bond2, convex_atom2 = convex_bond_atom(ms[i])
		#avail_atom_idx_2 = [idx for idx in convex_atom2 if atoms2[idx].IsInRingSize(6)]
		avail_atom_idx_2 = [idx for idx in convex_atom2 if atoms2[idx].IsInRingSize(6) or (atoms2[idx].IsInRingSize(5) and atoms2[idx].GetTotalNumHs() == 1)]
		avail_atom_idx_2 = np.array(avail_atom_idx_2) + len(atoms)
		index_check.append(list(avail_atom_idx_2))

		m_comb = Chem.CombineMols(seed,ms[i])
		linker = AllChem.MolFromSmiles(linker_sm)
		linker_indx = np.array([atom.GetIdx() for atom in linker.GetAtoms()]) + len(m_comb.GetAtoms())
		linker_indx = [int(l) for l in linker_indx]
		#m_comb2 = Chem.CombineMols(m_comb,linker)
		
		series = []
		series_energy = []
		series_mol = []
		print(len(avail_atom_idx_1)*len(avail_atom_idx_2))

		f1_ = open('collect_energy_smiles_%d_%d.txt' % (label, i),'w')
		f2_ = open('collect_structure_%d_%d.dump' % (label, i),'w')

		for a1 in avail_atom_idx_1:
			for a2 in avail_atom_idx_2:
				
				m_comb2 = Chem.CombineMols(m_comb,linker)
				edcombo = Chem.EditableMol(m_comb2)
				a1 = int(a1)
				a2 = int(a2)
				b1 = int(linker_indx[0])
				b2 = int(linker_indx[-1])
				series.append([a1,a2,b1,b2])
				
				add_crosslink[i] += 1
				for ci, cc in enumerate(index_check):
					if a1 in list(cc):
						add_crosslink[ci] += 1
				#print(add_crosslink)
		
				edcombo.AddBond(a1,b1,order=Chem.rdchem.BondType.SINGLE)
				edcombo.AddBond(a2,b2,order=Chem.rdchem.BondType.SINGLE)
				connected_m = edcombo.GetMol()
				connected_m = AllChem.AddHs(connected_m)
				atoms_m = connected_m.GetAtoms()
				hs_remove = []
				#print([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
				hs_a1 = max([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
				hs_a2 = max([n.GetIdx() for n in atoms_m[a2].GetNeighbors() if n.GetSymbol() == 'H'])
				if len(linker_indx) > 1:
					hs_b1 = max([n.GetIdx() for n in atoms_m[b1].GetNeighbors() if n.GetSymbol() == 'H'])
					hs_b2 = max([n.GetIdx() for n in atoms_m[b2].GetNeighbors() if n.GetSymbol() == 'H'])
					hs_b = [hs_b1,hs_b2]
				else:
					hs_b = [n.GetIdx() for n in atoms_m[b1].GetNeighbors() if n.GetSymbol() == 'H'][:2]
				hs_b.append(hs_a1)
				hs_b.append(hs_a2)
				hs_remove = list(sorted(hs_b,reverse=True))
				edcombo2 = Chem.EditableMol(connected_m)
				[ edcombo2.RemoveAtom(h_idx) for h_idx in hs_remove ]
				connected_m = edcombo2.GetMol()
				check = edcombo2.GetMol()
				

				#check rdmolops.RemoveStereochemistry()
				#heteroatom : ReplaceCore()
				params = AllChem.ETKDGv3()
				params.useSmallRingTorsions = True
				AllChem.EmbedMolecule(check,maxAttempts=5000, useRandomCoords=True, )
				AllChem.MMFFOptimizeMolecule(check, maxIters=1000)
				pdb_block = AllChem.MolToPDBBlock(check)
				pdb_file = open('molecule.pdb','w')
				pdb_file.write(pdb_block)
				pdb_file.close()

				energy, syms, positions, structure_str = ReaxFFminimize(pdb_file='molecule.pdb',path='./ReaxFF_min')
				series_energy.append(energy)
				series_mol.append(connected_m)
				f1_.write("%d %d %12.4f %s\n" % (a1,a2,energy,AllChem.MolToSmiles(connected_m)))
				f2_.write(structure_str)
				f1_.flush()
				f2_.flush()

		f1_.close()
		f2_.close()

		print(series)
		s_series_energy = [s-min(series_energy) for s in series_energy]
		print(s_series_energy)
		cind = s_series_energy.index(0)
		connected_m2 = series_mol[cind]

		#MMFFs_test(check)
		#check = AllChem.RemoveHs(check)
		#check = molwith_idx(check)
		#AllChem.Compute2DCoords(check)
		#Chem.rdCoordGen.AddCoords(check)
		#Chem.Draw.MolToFile(check,'tmp.png', size=(2000,2000),kekulize=True)
		#subprocess.call('imgcat tmp.png',shell=True)
		#Chem.Draw.MolToFile(check,'./%d_after.png' % i, size=(400,400),kekulize=True)#,highlightAtoms=avail_atom_idx_1, highlightBonds=convex_bond)
		seed = deepcopy(connected_m2)

		#for j in range(len(index_check)):
		#	ms_inds = index_check[j]
		#	if a1 in ms_inds:
		#		count[j] += 1
		MW = Descriptors.ExactMolWt(seed)
		print(MW)

	#inchi = AllChem.MolToInchi(connected_m)
	smi = AllChem.MolToSmiles(AllChem.RemoveHs(connected_m2))
	#final_mol_3d = MMFFs_3Dconstruct(connected_m)[0]
	#s = AllChem.MolToPDBBlock(final_mol_3d)
	return connected_m2, smi

	#Count the number of bonds added to each molecule	


# Revise
def Combine_Two(mol1,mol2, linker_sm='CCOCC', label=0):
	#ms is the list of mol objects
	#shuffle the molecules before combine
	#np.random.shuffle(ms)
	add_crosslink = np.zeros(2)
	seed = mol1
	index_check = []
	count = np.zeros(2)
	convex_bond, convex_atom = convex_bond_atom(seed)
	atoms = seed.GetAtoms()
	avail_atom_idx_0 = [idx for idx in convex_atom if atoms[idx].IsInRing()]
	index_check.append(avail_atom_idx_0)



	seed = AllChem.RemoveHs(seed)
	convex_bond, convex_atom = convex_bond_atom(seed)
	atoms = seed.GetAtoms()
	avail_atom_idx_1 = [idx for idx in convex_atom if atoms[idx].IsInRingSize(6) or (atoms[idx].IsInRingSize(5) and atoms[idx].GetTotalNumHs() == 1)]

	atoms2 = mol2.GetAtoms()
	convex_bond2, convex_atom2 = convex_bond_atom(mol2)
	avail_atom_idx_2 = [idx for idx in convex_atom2 if atoms2[idx].IsInRingSize(6) or (atoms2[idx].IsInRingSize(5) and atoms2[idx].GetTotalNumHs() == 1)]
	avail_atom_idx_2 = np.array(avail_atom_idx_2) + len(atoms)
	index_check.append(list(avail_atom_idx_2))

	m_comb = Chem.CombineMols(seed,mol2)
	linker = AllChem.MolFromSmiles(linker_sm)
	linker_indx = np.array([atom.GetIdx() for atom in linker.GetAtoms()]) + len(m_comb.GetAtoms())
	linker_indx = [int(l) for l in linker_indx]
	
	series = []
	#print(len(avail_atom_idx_1)*len(avail_atom_idx_2))
	mol_list = []
	smi_list = []

	for a1 in avail_atom_idx_1:
		for a2 in avail_atom_idx_2:				
			m_comb2 = Chem.CombineMols(m_comb,linker)
			edcombo = Chem.EditableMol(m_comb2)
			a1 = int(a1)
			a2 = int(a2)
			b1 = int(linker_indx[0])
			b2 = int(linker_indx[-1])
			series.append([a1,a2,b1,b2])
				
			edcombo.AddBond(a1,b1,order=Chem.rdchem.BondType.SINGLE)
			edcombo.AddBond(a2,b2,order=Chem.rdchem.BondType.SINGLE)
			connected_m = edcombo.GetMol()
			connected_m = AllChem.AddHs(connected_m)
			atoms_m = connected_m.GetAtoms()
			hs_remove = []
			hs_a1 = max([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
			hs_a2 = max([n.GetIdx() for n in atoms_m[a2].GetNeighbors() if n.GetSymbol() == 'H'])
			if len(linker_indx) > 1:
				hs_b1 = max([n.GetIdx() for n in atoms_m[b1].GetNeighbors() if n.GetSymbol() == 'H'])
				hs_b2 = max([n.GetIdx() for n in atoms_m[b2].GetNeighbors() if n.GetSymbol() == 'H'])
				hs_b = [hs_b1,hs_b2]
			else:
				hs_b = [n.GetIdx() for n in atoms_m[b1].GetNeighbors() if n.GetSymbol() == 'H'][:2]
			hs_b.append(hs_a1)
			hs_b.append(hs_a2)
			hs_remove = list(sorted(hs_b,reverse=True))
			edcombo2 = Chem.EditableMol(connected_m)
			[ edcombo2.RemoveAtom(h_idx) for h_idx in hs_remove ]
			connected_m = edcombo2.GetMol()
			check = edcombo2.GetMol()
			
			#check rdmolops.RemoveStereochemistry()
			#heteroatom : ReplaceCore()
			params = AllChem.ETKDGv3()
			params.useSmallRingTorsions = True
			check = Embedfrom2Dto3D(check)
			smi = AllChem.MolToSmiles(check)
			if smi not in smi_list:
				mol_list.append(check)
				smi_list.append(smi)

	return mol_list

	#Count the number of bonds added to each molecule	

# Developing...
def Crosslink_SMILES_C(mol1,mol2,connect='C'):

	combined_mol = []
	mol1 = AllChem.RemoveHs(mol1)
	if connect=='C':
		flag, mol1_list = Heteroatom_Func_Add_CH3_list(mol1)
	elif connect=='CC':
		flag, mol1_list = Heteroatom_Func_Add_CH2CH3_list(mol1)
	else:
		return False
	
	for mol1 in mol1_list:
		atoms = mol1.GetAtoms()
		a1idx = [a.GetIdx() for a in atoms][-1]
		a1sym = [a.GetSymbol() for a in atoms][-1]

		a1c, a1r = type_idx_sym(mol1)

		mol2 = AllChem.RemoveHs(mol2)
		a2c, a2r = type_idx_sym(mol2)
		pair_idx = []
		pair_symbol = []

		for a2_ in a2c.items():
			pair_idx.append([a1idx,a2_[0]])
			pair_symbol.append(a1sym+a2_[1])
		for a2_ in a2r.items():
			pair_idx.append([a1idx,a2_[0]])
			pair_symbol.append(a1sym+a2_[1])

		pair_new = []
		for pi, ps in zip(pair_idx, pair_symbol):
			pair_new.append(pi)
		pair_update = [[p[0],p[1]+len(atoms)] for p in pair_new] # update index for combined mol
		m_comb = Chem.CombineMols(mol1,mol2)

		for i, link in enumerate(pair_update):
			m_comb_dup = deepcopy(m_comb)
			a1 = link[0]
			a2 = link[1]
			edcombo = Chem.EditableMol(m_comb_dup)
			edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.SINGLE)
			connected_m = edcombo.GetMol()
			atoms_m = connected_m.GetAtoms()
			
			connected_m = AllChem.AddHs(connected_m)
			atoms_m = connected_m.GetAtoms()
			hs_remove = []
			#print(atoms_m[a1].GetSymbol(),atoms_m[a2].GetSymbol())
			#print([n.GetSymbol() for n in atoms_m[a1].GetNeighbors()],[n.GetSymbol() for n in atoms_m[a2].GetNeighbors()])
			hs_a1 = max([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
			hs_a2 = max([n.GetIdx() for n in atoms_m[a2].GetNeighbors() if n.GetSymbol() == 'H'])
			hs_remove.append(hs_a1)
			hs_remove.append(hs_a2)

			hs_remove = list(sorted(set(hs_remove), reverse=True))
			edcombo2 = Chem.EditableMol(connected_m)
			[ edcombo2.RemoveAtom(h_idx) for h_idx in hs_remove ]
			connected_m = edcombo2.GetMol()
			final = deepcopy(connected_m)
			MW = Descriptors.ExactMolWt(final)
			try:
				final2 = AllChem.RemoveHs(final)
				combined_mol.append(final2)
			except:
				continue

	mols = [[mol_,0] for mol_ in combined_mol]
	return mols


def Crosslink_withC(mol1,mol2,connect='C'):
	mols = Crosslink_SMILES_C(mol1,mol2,connect)

	mols_arr = np.array(mols)[:,0]
	mols_dict = {}
	for i, mol in enumerate(mols_arr):
		mol_noH = AllChem.RemoveHs(mol)
		smi = AllChem.MolToSmiles(mol_noH,canonical=True, isomericSmiles=False)
		mols_dict[smi] = mol
	Combined_mols = [AllChem.MolFromSmiles(smi) for smi in mols_dict.keys()]

	return Combined_mols