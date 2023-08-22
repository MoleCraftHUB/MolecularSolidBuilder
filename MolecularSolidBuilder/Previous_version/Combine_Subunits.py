import numpy as np
import os, sys, glob, subprocess
from itertools import combinations
from copy import deepcopy
from ase.io import read, write
from ase import Atoms, Atom
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from Build_HC_revise import convex_bond_atom
from Run_MD import ReaxFFminimize 


from ase.visualize import view
from Run_MD import ReaxFFminimize

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

def Combine_Random(ms, linker_sm='CCOCC'):

	#ms is the list of mol objects
	#shuffle the molecules before combine
	if len(ms) == 1:
		#smi = AllChem.MolToSmiles(AllChem.RemoveHs(ms[0]))
		smi = None
		#final_mol_3d = MMFFs_3Dconstruct(connected_m)[0]
		#s = AllChem.MolToPDBBlock(final_mol_3d)
		new_mol = AllChem.AddHs(ms[0])
		return new_mol, smi

	elif len(ms) > 1:
		np.random.shuffle(ms)

		add_crosslink = np.zeros(len(ms))
		seed = ms[0]
		seed = AllChem.RemoveHs(seed)
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
			avail_atom_idx_1 = [idx for idx in convex_atom if (atoms[idx].IsInRingSize(6) or atoms[idx].IsInRingSize(5)) and atoms[idx].GetTotalNumHs() == 1]
			
			ms[i] = AllChem.RemoveHs(ms[i])
			#check1 = deepcopy(ms[i])
			#check1 = AllChem.RemoveHs(check1)
			#AllChem.Compute2DCoords(ms[i])
			#Chem.rdCoordGen.AddCoords(check1)
			#Chem.Draw.MolToFile(ms[i],'./test.png' , size=(200,200))#,kekulize=True,highlightAtoms=avail_atom_idx_1)#, highlightBonds=convex_bond)
			#subprocess.call('imgcat ./test.png' ,shell=True)
			atoms2 = ms[i].GetAtoms()
			convex_bond2, convex_atom2 = convex_bond_atom(ms[i])
			avail_atom_idx_2 = [idx for idx in convex_atom2 if (atoms2[idx].IsInRingSize(6) or atoms2[idx].IsInRingSize(5)) and atoms2[idx].GetTotalNumHs() == 1]
			avail_atom_idx_2 = np.array(avail_atom_idx_2) + len(atoms)
			index_check.append(list(avail_atom_idx_2))

			m_comb = Chem.CombineMols(seed,ms[i])
			linker = AllChem.MolFromSmiles(linker_sm)
			linker_indx = np.array([atom.GetIdx() for atom in linker.GetAtoms()]) + len(m_comb.GetAtoms())
			linker_indx = [int(l) for l in linker_indx]
			m_comb2 = Chem.CombineMols(m_comb,linker)
			edcombo = Chem.EditableMol(m_comb2)

			test_ = deepcopy(avail_atom_idx_1)
			flag = True
			while flag:
				np.random.shuffle(test_)
				a1 = int(test_[0])
				flag_tmp = [ True for ci, cc in enumerate(index_check) if (a1 in cc) and add_crosslink[ci] <= 2 ]
				flag = not any(flag_tmp)

			a2 = int(np.random.choice(avail_atom_idx_2, size=1)[0])
			b1 = int(linker_indx[0])
			b2 = int(linker_indx[-1])
			#print(index_check, a1, a2)
			
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
			print([n.GetIdx() for n in atoms_m[a1].GetNeighbors() if n.GetSymbol() == 'H'])
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

			#MMFFs_test(check)
			#check = AllChem.RemoveHs(check)
			#check = molwith_idx(check)
			#AllChem.Compute2DCoords(check)
			#Chem.rdCoordGen.AddCoords(check)
			#Chem.Draw.MolToFile(check,'tmp.png', size=(2000,2000),kekulize=True)
			#subprocess.call('imgcat tmp.png',shell=True)
			#Chem.Draw.MolToFile(check,'./%d_after.png' % i, size=(400,400),kekulize=True)#,highlightAtoms=avail_atom_idx_1, highlightBonds=convex_bond)
			seed = deepcopy(connected_m)
			for j in range(len(index_check)):
				ms_inds = index_check[j]
				if a1 in ms_inds:
					count[j] += 1
			MW = Descriptors.ExactMolWt(seed)
			print(MW)

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

		#sys.exit()

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
