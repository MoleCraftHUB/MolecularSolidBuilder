import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from .Utility import *
from ase.io import read, write
import numpy as np
import os, sys, glob, subprocess, copy
from itertools import combinations
import itertools

def Intramolecular_Bond(mol):
	test_mols = []
	test_info = []

	mol = AllChem.RemoveHs(mol)
	atoms = mol.GetAtoms()
	idx_wH = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() > 0]
	nconnect_idxs = []
	h_num_checks = []
	for i in range(len(idx_wH)):
		idx1 = idx_wH[i]
		for j in range(i+1,len(idx_wH)):
			idx2 = idx_wH[j]
			connect_idx = AllChem.GetShortestPath(mol,idx1,idx2)
			Dist3D = AllChem.Get3DDistanceMatrix(mol)
			check_bonded = mol.GetBondBetweenAtoms(idx1,idx2)
			rf = mol.GetRingInfo()
			arf = rf.AtomRings()
			check_ih = [atoms[iid].GetTotalNumHs() for iid in connect_idx]
			check_ir = [atoms[iid].IsInRing() for iid in connect_idx]
			if (Dist3D[idx1,idx2] < 4) and (check_bonded==None) and (not rf.AreAtomsInSameRing(idx1,idx2)):
				mol_copy = copy.deepcopy(mol)
				atoms_copy = mol_copy.GetAtoms()
				#print(idx1,idx2,Dist3D[idx1,idx2],check_bonded)
				#print(atoms_copy[idx1].GetNumExplicitHs(),atoms_copy[idx2].GetNumExplicitHs())

				edcombo = Chem.EditableMol(mol_copy)
				edcombo.AddBond(idx1,idx2,order=Chem.rdchem.BondType.SINGLE)
				back = edcombo.GetMol()

				atoms2 = back.GetAtoms()
				atoms2[idx1].SetNumExplicitHs(0)
				atoms2[idx2].SetNumExplicitHs(0)

				back_H = AllChem.AddHs(back,addCoords=True)
				atoms2_H = back_H.GetAtoms()
				test1 = [n.GetIdx() for n in atoms2_H[idx1].GetNeighbors() if n.GetSymbol() == 'H']
				test2 = [n.GetIdx() for n in atoms2_H[idx2].GetNeighbors() if n.GetSymbol() == 'H']
				em2 = Chem.EditableMol(back_H)
				atomsToRemove = sorted(test1+test2,reverse=True)
				for idd in atomsToRemove:
					em2.RemoveAtom(idd)
				rem_H = em2.GetMol()
				AllChem.Kekulize(rem_H)
				rf2 = rem_H.GetRingInfo()
				arf2 = rf2.AtomRings()
				size_ring = len([c for c in arf2 if len(c) < 5]) + len([c for c in arf2 if len(c) > 7])
				flag = 0
				for c1, c2 in zip(check_ih[1:-1],check_ir[1:-1]):
					if c1 > 0 and c2 == True:
						flag += 1

				if size_ring == 0 and flag == 0:
					print(check_ih[1:-1],check_ir[1:-1])
					
					test_mols.append(rem_H)
					test_info.append("Bonding %s %s" % (idx1,idx2))

	return test_mols, test_info

def arylbond_combine(mol1,mol2):
	
	combined_mols = []
	mol1 = AllChem.RemoveHs(mol1)
	atoms1 = mol1.GetAtoms()

	mol2 = AllChem.RemoveHs(mol2)
	atoms2 = mol2.GetAtoms()

	v1 = Find_Vertex_v2(mol1)
	v2 = Find_Vertex_v2(mol2)

	v1_aryl = [v for v in v1 if len(v) <= 4]
	v2_aryl = [v for v in v2 if len(v) <= 4]

	pair_six  = []
	pair_five = []
	for i, v_1 in enumerate(v1_aryl):
		for j, v_2 in enumerate(v2_aryl):
			if len(v_1) + len(v_2) == 6:
				pair_six.append([v_1,v_2])
				v_2_r = v_2[::-1]
				pair_six.append([v_1,v_2_r])
			if len(v_1) + len(v_2) == 5:
				pair_five.append([v_1,v_2])
				v_2_r = v_2[::-1]
				pair_five.append([v_1,v_2_r])

	## Six ring
	pair_six_reduce = []
	for i, p6 in enumerate(pair_six):
		pv1 = p6[0]
		pv2 = p6[1]
		pv1_h = [atoms1[idx].GetTotalNumHs() for idx in pv1]
		pv2_h = [atoms2[idx].GetTotalNumHs() for idx in pv2]
		if (pv1_h[0]==pv2_h[0]) and (pv1_h[-1]==pv2_h[-1]):
			pair_six_reduce.append([pv1,pv2])

	m_comb = Chem.CombineMols(mol1,mol2)
	for i, p6 in enumerate(pair_six_reduce):
		m_comb_dup = copy.deepcopy(m_comb)	
		atoms_comb = m_comb_dup.GetAtoms()
		pv1 = p6[0]
		pv2 = [p+len(atoms2) for p in p6[1]]
		a1 = pv1[0]
		a2 = pv2[0]
		b1 = pv1[-1]
		b2 = pv2[-1]
		m_comb_dup = AllChem.AddHs(m_comb_dup)
		AllChem.Kekulize(m_comb_dup)

		atoms_comb2 = m_comb_dup.GetAtoms()
		edcombo = Chem.EditableMol(m_comb_dup)
		#Remove hydrogens
		hs_remove = []
		hs_remove_count = []
		for idx in [a1,a2,b1,b2]:
			h_idx = [n.GetIdx() for n in atoms_comb2[idx].GetNeighbors() if n.GetSymbol() == 'H']
			hs_remove += h_idx
			hs_remove_count.append(len(h_idx))
		hs_remove = list(sorted(set(hs_remove), reverse=True))
		[ edcombo.RemoveAtom(h_idx) for h_idx in hs_remove ]
		connected_m = edcombo.GetMol()
		connected_m = AllChem.RemoveHs(connected_m)
		#Plot_2Dmol(connected_m,pngfilename='test_6_%d.png' % i)
		
		if hs_remove_count[0] == 1 and hs_remove_count[1] == 1:
			edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.SINGLE)
		elif hs_remove_count[0] == 2 and hs_remove_count[1] == 2:
			edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.DOUBLE)
		if hs_remove_count[2] == 1 and hs_remove_count[3] == 1:
			edcombo.AddBond(b1,b2,order=Chem.rdchem.BondType.SINGLE)
		elif hs_remove_count[2] == 2 and hs_remove_count[3] == 2:
			edcombo.AddBond(b1,b2,order=Chem.rdchem.BondType.DOUBLE)
		connected_m = edcombo.GetMol()
		connected_m = AllChem.RemoveHs(connected_m)
		
		#Plot_2Dmol(connected_m,pngfilename='test_6_%d.png' % i)
		#sys.exit()
		#atoms_m = connected_m.GetAtoms()
		final = copy.deepcopy(connected_m)
		combined_mols.append(final)
		

	"""
	## Five ring
	pair_five_reduce = []
	for i, p5 in enumerate(pair_five):
		pv1 = p5[0]
		pv2 = p5[1]
		pv1_h = [atoms1[idx].GetTotalNumHs() for idx in pv1]
		pv2_h = [atoms2[idx].GetTotalNumHs() for idx in pv2]
		if (pv1_h[0]==pv2_h[0]) and (pv1_h[-1]==pv2_h[-1]):
			pair_five_reduce.append([pv1,pv2])
		elif (pv1_h[0]!=pv2_h[0]) and (pv1_h[-1]!=pv2_h[-1]) and (pv1_h[0]==pv2_h[-1]) and (pv1_h[-1]==pv2_h[0]):
			pv2.reverse()
			pair_five_reduce.append([pv1,pv2])

	m_comb = Chem.CombineMols(mol1,mol2)
	for i, p5 in enumerate(pair_five_reduce):
		m_comb_dup = copy.deepcopy(m_comb)
		pv1 = p5[0]
		pv2 = [p+len(atoms2) for p in p5[1]]
		a1 = pv1[0]
		a2 = pv2[0]
		b1 = pv1[-1]
		b2 = pv2[-1]
		edcombo = Chem.EditableMol(m_comb_dup)
		if atoms_comb[a1].GetTotalNumHs()==1 and atoms_comb[a2].GetTotalNumHs() == 1:
			edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.SINGLE)
		elif atoms_comb[a1].GetTotalNumHs()==2 and atoms_comb[a2].GetTotalNumHs() == 2:
			edcombo.AddBond(a1,a2,order=Chem.rdchem.BondType.DOUBLE)
		if atoms_comb[b1].GetTotalNumHs()==1 and atoms_comb[b2].GetTotalNumHs() == 1:
			edcombo.AddBond(b1,b2,order=Chem.rdchem.BondType.SINGLE)
		elif atoms_comb[b1].GetTotalNumHs()==2 and atoms_comb[b2].GetTotalNumHs() == 2:
			edcombo.AddBond(b1,b2,order=Chem.rdchem.BondType.DOUBLE)
		connected_m = edcombo.GetMol()
		connected_m = AllChem.AddHs(connected_m)

		atoms_m = connected_m.GetAtoms()
		
		#Remove hydrogens
		hs_remove = []
		for idx in [a1,a2,b1,b2]:
			h_idx = max([n.GetIdx() for n in atoms_m[idx].GetNeighbors() if n.GetSymbol() == 'H'])
			hs_remove.append(h_idx)
		hs_remove = list(sorted(set(hs_remove), reverse=True))
		edcombo2 = Chem.EditableMol(connected_m)
		[ edcombo2.RemoveAtom(h_idx) for h_idx in hs_remove ]
		connected_m = edcombo2.GetMol()
		#Plot_2Dmol(connected_m,pngfilename='test_5_%d.png' % i)
		final = copy.deepcopy(connected_m)
		combined_mols.append(final)
	"""
	combined_mols_smi = [AllChem.MolToSmiles(mol) for mol in combined_mols]
	combined_mols_smi_set = list(set(combined_mols_smi))
	print(combined_mols_smi_set)
	combined_mols_reduced = [AllChem.MolFromSmiles(smi) for smi in combined_mols_smi_set]
	return combined_mols_reduced


def molwith_idx(mol):
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(atom.GetIdx())
	return mol

def molwith_hyb(mol):
	for atom in mol.GetAtoms():
		if atom.GetHybridization() == AllChem.HybridizationType.SP2:
			atom.SetAtomMapNum(2)
		elif atom.GetHybridization() == AllChem.HybridizationType.SP3:
			atom.SetAtomMapNum(3)
	return mol

def isRingAromatic(mol, bondRing):
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
        return True

def Npi_Aromaticity(Ncarbon):
	N = (Ncarbon-2) % 4
	if N == 0:
		return True
	else:
		return False

def convex_bond_atom(m):
	
	ring  = m.GetRingInfo()
	ring_members = ring.AtomRings()

	#Flatten the list of rings
	rmem = []
	for rm in ring_members:
		for t in rm:
			rmem.append(t)
	rcount = {}
	for e in rmem:
		rcount[e] = rmem.count(e)

	convex_bond = []
	convex_bond_idx = []
	for idx, bond in enumerate(m.GetBonds()):
		c1 = bond.GetBeginAtom()
		c2 = bond.GetEndAtom()
		n1 = [n.GetIdx() for n in c1.GetNeighbors()]
		n2 = [n.GetIdx() for n in c2.GetNeighbors()]
		
		if c1.GetIdx() in rmem: c1_rct = rcount[c1.GetIdx()]
		else: c1_rct = 0
		if c2.GetIdx() in rmem: c2_rct = rcount[c2.GetIdx()]
		else: c2_rct = 0

		flag = [c1.GetIdx() in ring_m and c2.GetIdx() in ring_m for ring_m in ring_members].count(True)
		if c1_rct < 3 and c2_rct < 3 and flag != 2:
			convex_bond.append(bond)
			convex_bond_idx.append(idx)
	
	convex_atom = []
	convex_atom_idx = []
	atoms = m.GetAtoms()
	ExH = [atom.GetTotalNumHs() for atom in atoms]
	for bond in convex_bond:
		a = bond.GetBeginAtom()
		b = bond.GetEndAtom()
		aidx = bond.GetBeginAtomIdx()
		bidx = bond.GetEndAtomIdx()
		if ExH[aidx] > 0 and a.IsInRing():
			convex_atom.append(a)
			convex_atom_idx.append(aidx)
		if ExH[bidx] > 0 and b.IsInRing():
			convex_atom.append(b)
			convex_atom_idx.append(bidx)
		convex_atom_idx = list(set(convex_atom_idx))

	return convex_bond_idx, convex_atom_idx

def Find_Vertex_v2(m):

	#m should have explicit hydrogens
	m = AllChem.AddHs(m)
	m = AllChem.RemoveHs(m)

	convex_bond, convex_atom = convex_bond_atom(m)
	
	grow_index = []
	atoms = m.GetAtoms()
	bonds = m.GetBonds()

	for bond_idx in convex_bond:
		bond = bonds[bond_idx]
		t1 = bond.GetBeginAtomIdx()
		t2 = bond.GetEndAtomIdx()
		t1_atom = bond.GetBeginAtom()
		t2_atom = bond.GetEndAtom()
		#Directly bonded
		if t1 in convex_atom and t2 in convex_atom:
			idxs = [t1,t2]
			grow_index.append(idxs)
		#One atom in the middle
		if t1 in convex_atom and t2 not in convex_atom:
			bond2 = t2_atom.GetBonds()
			bb = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			if len(bb) > 0:
				tt1 = bb[0].GetBeginAtomIdx()
				tt2 = bb[0].GetEndAtomIdx()
				if tt1 != t1 and tt1 != t2 and tt1 in convex_atom:
					idxs = [t1,t2,tt1]
					if [tt1,t2,t1] not in grow_index:
						grow_index.append(idxs)
				if tt2 != t1 and tt2 != t2 and tt2 in convex_atom:
					idxs = [t1,t2,tt2]
					if [tt2,t2,t1] not in grow_index:
						grow_index.append(idxs)
		if t1 not in convex_atom and t2 in convex_atom:
			bond2 = t1_atom.GetBonds()
			bb = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			if len(bb) > 0:
				tt1 = bb[0].GetBeginAtomIdx()
				tt2 = bb[0].GetEndAtomIdx()
				if tt1 != t1 and tt1 != t2 and tt1 in convex_atom:
					idxs = [t2,t1,tt1]
					if [tt1,t1,t2] not in grow_index:
						grow_index.append(idxs)
				if tt2 != t1 and tt2 != t2 and tt2 in convex_atom:
					idxs = [t2,t1,tt2]
					if [tt2,t1,t2] not in grow_index:
						grow_index.append(idxs)

		#Two atom in the middle
		if t1 not in convex_atom and t2 not in convex_atom:
			if all([True if n.IsInRing() else False for n in t1_atom.GetNeighbors() ]) and all([True if n.IsInRing() else False for n in t2_atom.GetNeighbors() ]):
				bond1 = t1_atom.GetBonds()
				bond2 = t2_atom.GetBonds()
				bb1 = [b for b in bond1 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
				bb2 = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
				if len(bb1) > 0 and len(bb2) > 0:
					bond1_idx = bb1[0].GetIdx()
					bond2_idx = bb2[0].GetIdx()
					tt1_ = [bb1[0].GetBeginAtomIdx(), bb1[0].GetEndAtomIdx()]
					tt1_.remove(t1)
					tt2_ = [bb2[0].GetBeginAtomIdx(), bb2[0].GetEndAtomIdx()]
					tt2_.remove(t2)
					tmp = tt1_ + tt2_
					tmp2 = [s for s in tmp if s in convex_atom]
					if len(tmp2) == 2:
						idxs = [tmp2[0]] + [t1,t2] + [tmp2[1]]
						grow_index.append(idxs)

					#Four atom in the middle
					if (tt1_[0] not in convex_atom) and (tt2_[0] not in convex_atom):
						bondn1 = atoms[tt1_[0]].GetBonds()
						bondn2 = atoms[tt2_[0]].GetBonds()
						bbb1 = [b for b in bondn1 if b.GetIdx() != bond1_idx and b.GetIdx() in convex_bond]
						bbb2 = [b for b in bondn2 if b.GetIdx() != bond2_idx and b.GetIdx() in convex_bond]
						if len(bbb1) > 0 and len(bbb2) > 0:
							ttt1_ = [bbb1[0].GetBeginAtomIdx(), bbb1[0].GetEndAtomIdx()]
							ttt2_ = [bbb2[0].GetBeginAtomIdx(), bbb2[0].GetEndAtomIdx()]
							ttt1_.remove(tt1_[0])
							ttt2_.remove(tt2_[0])

							if (ttt1_[0] in convex_atom) and (ttt2_[0] in convex_atom):
								idxs = ttt1_ + tt1_ + [t1,t2] + tt2_ + ttt2_
								grow_index.append(idxs)
						
		if t1 not in convex_atom and t2 not in convex_atom:
			bond1 = t1_atom.GetBonds()
			bond2 = t2_atom.GetBonds()
			bb1 = [b for b in bond1 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			bb2 = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			if len(bb1) > 0 and len(bb2) > 0:
				bond1_idx = bb1[0].GetIdx()
				bond2_idx = bb2[0].GetIdx()
				tt1_ = [bb1[0].GetBeginAtomIdx(), bb1[0].GetEndAtomIdx()]
				tt1_.remove(t1)
				tt2_ = [bb2[0].GetBeginAtomIdx(), bb2[0].GetEndAtomIdx()]
				tt2_.remove(t2)
				tmp = tt1_ + tt2_
				tmp2 = [s for s in tmp if s in convex_atom]
				#Three atom in the middle
				if (tt1_[0] not in convex_atom) and (tt2_[0] in convex_atom):
					bondn1 = atoms[tt1_[0]].GetBonds()
					bbb1 = [b for b in bondn1 if b.GetIdx() != bond1_idx and b.GetIdx() in convex_bond]
					if len(bbb1) > 0:
						ttt1_ = [bbb1[0].GetBeginAtomIdx(), bbb1[0].GetEndAtomIdx()]
						ttt1_.remove(tt1_[0])
						if (ttt1_[0] in convex_atom):
							idxs = ttt1_ + tt1_ + [t1,t2] + tt2_
							grow_index.append(idxs)
				elif (tt1_[0] in convex_atom) and (tt2_[0] not in convex_atom):
					bondn1 = atoms[tt2_[0]].GetBonds()
					bbb1 = [b for b in bondn1 if b.GetIdx() != bond1_idx and b.GetIdx() in convex_bond]
					if len(bbb1) > 0:
						ttt2_ = [bbb1[0].GetBeginAtomIdx(), bbb1[0].GetEndAtomIdx()]
						ttt2_.remove(tt2_[0])
						if (ttt2_[0] in convex_atom):
							idxs = ttt2_ + tt2_ + [t2,t1] + tt1_
							idxs_test = tt1_ + [t1,t2] + tt2_ + ttt2_
							if idxs_test not in grow_index:
								grow_index.append(idxs)

	grow_index = sorted(grow_index,key=lambda x:len(x),reverse=True)

	for i, each in enumerate(grow_index):
		if len(each) == 6:
			test_list = grow_index[i+1:]
			for j, test in enumerate(test_list):
				if each[0] in test or each[-1] in test:
					grow_index.remove(test)
	
	grow_index = [each for each in grow_index \
		          if all([True if atoms[idx].GetSymbol() =='C' else False for idx in each])]
	return grow_index


def Propagate_v2(main,vertx,six_ring=False,five_ring=False,nbfive=0):

	new_mols = []
	new_mols_inchi = []
	#Assume the main mol is aromatic preserving the role of 4N+2 pi electrons
	#Each carbon has a single pi electron
	ring  = main.GetRingInfo()
	ring_members = ring.AtomRings()

	for v in vertx:
		mols = []
		if len(v) == 2: #case1
			if six_ring and not five_ring:		
				sms = ["[CH1][CH1][CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1][CH1][CH1]"]
			else:
				sms = ["[CH1][CH1][CH1][CH1]","[CH1][CH1][CH1]"]
		elif len(v) == 3: #case2
			if six_ring and not five_ring:
				sms = ["[CH1][CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1][CH1]"]
			else:
				sms = ["[CH1][CH1][CH1]","[CH1][CH1]"]
		elif len(v) == 4: #case3
			if six_ring and not five_ring:
				sms = ["[CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1]"]
			else:
				sms = ["[CH1][CH1]","[CH1]"]
		elif len(v) == 5: #case4
			if six_ring and not five_ring:
				sms = ["[CH1]"]
			else:
				sms = []
		elif len(v) == 6: #case5
			sms = []

		if len(sms) > 0 and len(v) < 6:
			for typ in range(len(sms)):
				frg = AllChem.MolFromSmiles(sms[typ])
				#find how many of fragment carbons should be SP2 hybridization
				main_atoms = main.GetAtoms()
				frg_atoms = frg.GetAtoms()

				main_Ncarbons = len([atom for atom in main_atoms if atom.GetSymbol() == 'C'])
				frg_Ncarbons = len([atom for atom in frg_atoms if atom.GetSymbol() == 'C'])
				new_Npi = [i for i in range(frg_Ncarbons,-1,-1) if Npi_Aromaticity(i+main_Ncarbons)]

				mcomb = Chem.CombineMols(main,frg)
				mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
				mcomb_atoms = mcomb.GetAtoms()

				mcomb = AllChem.AddHs(mcomb)
				main_idx = mcomb_idx[:-len(frg.GetAtoms())]
				frg_idx = mcomb_idx[-len(frg.GetAtoms()):]

				#atoms to make bonding with frag
				ringmem = [[len(r) for r in ring_members if el in r] for el in v]
				count_five = [rmi.count(5) for rmi in ringmem]
				if (len(frg_atoms)+len(v)==6 and any(count_five)>=0) or (len(frg_atoms)+len(v)==5 and any(count_five)<=nbfive):
					edcombo = Chem.EditableMol(mcomb)
					if (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.DOUBLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					else:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)

					ht = []
					for vi in range(len(v)):
						hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
						if len(hs) > 0:
							ht += hs
					ht = sorted(ht,reverse=True)
					[edcombo.RemoveAtom(t) for t in ht]

					fm = edcombo.GetMol()
					AllChem.Kekulize(fm)

					fm = AllChem.RemoveHs(fm)
					atoms2 = fm.GetAtoms()
					bonds2 = fm.GetBonds()
					[atom.SetNumRadicalElectrons(0) for atom in atoms2]

					for ll, atom in enumerate(atoms2):
						bts = []
						for b in atom.GetBonds():
							bts.append(b.GetBondTypeAsDouble())
						valence = sum(bts)+atom.GetTotalNumHs()
						atom.SetNumRadicalElectrons(int(4.5-valence))
					[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]
					
					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 3 and len(bnn) == 2) or (len(ann) == 2 and len(bnn) == 3):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 2 and len(bnn) == 2):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					[atom.SetNumExplicitHs(2) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]
					[atom.SetNumRadicalElectrons(0) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]

					smi = AllChem.MolToSmiles(fm)#,kekuleSmiles=True)
					inchi= AllChem.MolToInchi(fm) 

					#get all resulting combined molecules
					#get InChI output string -> store in the single list..
					new_mols_inchi.append(inchi)

		elif len(v) == 5 and len(sms) == 0:
			#print("make connection")
			atoms = main.GetAtoms()
			main2 = AllChem.AddHs(main)
			edcombo = Chem.EditableMol(main2)
			connect = False
			if (atoms[v[0]].GetTotalNumHs()) == 2 and (atoms[v[-1]].GetTotalNumHs()) == 2:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.DOUBLE)
				connect = True
			elif (atoms[v[0]].GetTotalNumHs()) == 1 and (atoms[v[-1]].GetTotalNumHs()) == 1:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.SINGLE)
				connect = True

			if connect:
				ht = []
				for vi in range(len(v)):
					hs = sorted([n.GetIdx() for n in main2.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
					if len(hs) > 0:
						ht += hs
				ht = sorted(ht,reverse=True)
				[edcombo.RemoveAtom(t) for t in ht]
				fm = edcombo.GetMol()
				AllChem.Kekulize(fm)
				fm = AllChem.RemoveHs(fm)
				atoms2 = fm.GetAtoms()
				bonds2 = fm.GetBonds()
				
				[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]

				for bi in range(len(bonds2)):
					bond = bonds2[bi]
					a=bond.GetBeginAtom()
					b=bond.GetEndAtom()
					ah=a.GetHybridization()
					bh=b.GetHybridization()
					if ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3:
						bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
						a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						a.UpdatePropertyCache()
						b.UpdatePropertyCache()
						bonds2 = fm.GetBonds()
				[atom.SetNumRadicalElectrons(0) for atom in atoms2]
				smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
				inchi= AllChem.MolToInchi(fm) 

				#get all resulting combined molecules
				#get InChI output string -> store in the single list..
				new_mols_inchi.append(inchi)

		if len(v) == 6 and len(sms) == 0:
			atoms = main.GetAtoms()
			main2 = AllChem.AddHs(main)
			edcombo = Chem.EditableMol(main2)
			connect = False
			if (atoms[v[0]].GetTotalNumHs()) == 2 and (atoms[v[-1]].GetTotalNumHs()) == 2:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.DOUBLE)
				connect = True
			elif (atoms[v[0]].GetTotalNumHs()) == 1 and (atoms[v[-1]].GetTotalNumHs()) == 1:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.SINGLE)
				connect = True

			if connect:
				ht = []
				for vi in range(len(v)):
					hs = sorted([n.GetIdx() for n in main2.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
					if len(hs) > 0:
						ht += hs
				ht = sorted(ht,reverse=True)
				[edcombo.RemoveAtom(t) for t in ht]
				fm = edcombo.GetMol()

				AllChem.Kekulize(fm)
				fm = AllChem.RemoveHs(fm)
				atoms2 = fm.GetAtoms()
				bonds2 = fm.GetBonds()
				
				[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]

				for bi in range(len(bonds2)):
					bond = bonds2[bi]
					a=bond.GetBeginAtom()
					b=bond.GetEndAtom()
					ah=a.GetHybridization()
					bh=b.GetHybridization()
					if ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3:
						bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
						a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						a.UpdatePropertyCache()
						b.UpdatePropertyCache()
						bonds2 = fm.GetBonds()
				[atom.SetNumRadicalElectrons(0) for atom in atoms2]
				smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
				inchi= AllChem.MolToInchi(fm) 

				#get all resulting combined molecules
				#get InChI output string -> store in the single list..
				new_mols_inchi.append(inchi)

	new_mols_inchi = list(set(new_mols_inchi))
	new_mols = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	return new_mols

def Propagate_fillup4(main,vertx,six_ring=False,five_ring=False,nbfive=1):

	new_mols = []
	new_mols_inchi = []
	#Assume the main mol is aromatic preserving the role of 4N+2 pi electrons
	#Each carbon has a single pi electron
	ring  = main.GetRingInfo()
	ring_members = ring.AtomRings()

	#main_idx = [atom.GetIdx() for atom in main.GetAtoms()]
	for v in vertx:
		mols = []
		if len(v) == 2: #case1
			if six_ring and not five_ring:		
				sms = ["[CH1][CH1][CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1][CH1][CH1]"]
			else:
				sms = ["[CH1][CH1][CH1][CH1]","[CH1][CH1][CH1]"]
		elif len(v) == 3: #case2
			if six_ring and not five_ring:
				sms = ["[CH1][CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1][CH1]"]
			else:
				sms = ["[CH1][CH1][CH1]","[CH1][CH1]"]
		elif len(v) == 4: #case3
			if six_ring and not five_ring:
				sms = ["[CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1]"]
			else:
				sms = ["[CH1][CH1]","[CH1]"]
		elif len(v) == 5: #case4
			if six_ring and not five_ring:
				sms = ["[CH1]"]
			else:
				sms = []
		elif len(v) == 6: #case5
			sms = []

		if len(sms) > 0 and len(v) == 4:
			#print("add rings")
			for typ in range(len(sms)):
				frg = AllChem.MolFromSmiles(sms[typ])
				#find how many of fragment carbons should be SP2 hybridization
				main_atoms = main.GetAtoms()
				frg_atoms = frg.GetAtoms()

				main_Ncarbons = len([atom for atom in main_atoms if atom.GetSymbol() == 'C'])
				frg_Ncarbons = len([atom for atom in frg_atoms if atom.GetSymbol() == 'C'])
				new_Npi = [i for i in range(frg_Ncarbons,-1,-1) if Npi_Aromaticity(i+main_Ncarbons)]

				mcomb = Chem.CombineMols(main,frg)
				mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
				mcomb_atoms = mcomb.GetAtoms()

				mcomb = AllChem.AddHs(mcomb)
				main_idx = mcomb_idx[:-len(frg.GetAtoms())]
				frg_idx = mcomb_idx[-len(frg.GetAtoms()):]

				#atoms to make bonding with frag
				ringmem = [[len(r) for r in ring_members if el in r] for el in v]
				count_five = [rmi.count(5) for rmi in ringmem]
				if any(count_five) <= nbfive:
					edcombo = Chem.EditableMol(mcomb)
					if (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						continue
						#if len(v) > 1:
						#	edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.DOUBLE)
						#	edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.DOUBLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					#elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
					#	edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
					#	edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					else:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)

					ht = []
					for vi in range(len(v)):
						hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
						if len(hs) > 0:
							ht += hs
					ht = sorted(ht,reverse=True)
					[edcombo.RemoveAtom(t) for t in ht]

					fm = edcombo.GetMol()
					AllChem.Kekulize(fm)
					atoms2 = fm.GetAtoms()
					bonds2 = fm.GetBonds()
					fm = AllChem.RemoveHs(fm)
					[atom.SetNumRadicalElectrons(0) for atom in fm.GetAtoms()]

					for ll, atom in enumerate(atoms2):
						bts = []
						for b in atom.GetBonds():
							bts.append(b.GetBondTypeAsDouble())
						valence = sum(bts)+atom.GetTotalNumHs()
						atom.SetNumRadicalElectrons(int(4.5-valence))
					[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]
					#
					#fm = molwith_idx(fm)

					
					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 3 and len(bnn) == 2) or (len(ann) == 2 and len(bnn) == 3):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 2 and len(bnn) == 2):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					[atom.SetNumExplicitHs(2) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]
					[atom.SetNumRadicalElectrons(0) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]

					smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
					inchi= AllChem.MolToInchi(fm) 

					#get all resulting combined molecules
					#get InChI output string -> store in the single list..
					new_mols_inchi.append(inchi)
		elif len(v) == 6:
			main = AllChem.AddHs(main)
			edcombo = Chem.EditableMol(main)
			edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.SINGLE)
			ht = []
			for vi in range(len(v)):
				hs = sorted([n.GetIdx() for n in main.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
				if len(hs) > 0:
					ht += hs
			ht = sorted(ht,reverse=True)
			[edcombo.RemoveAtom(t) for t in ht]
			fm = edcombo.GetMol()
			AllChem.Kekulize(fm)
			fm = AllChem.RemoveHs(fm)
			atoms2 = fm.GetAtoms()
			bonds2 = fm.GetBonds()
			
			[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]

			for bi in range(len(bonds2)):
				bond = bonds2[bi]
				a=bond.GetBeginAtom()
				b=bond.GetEndAtom()
				ah=a.GetHybridization()
				bh=b.GetHybridization()
				if ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3:
					bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
					a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
					b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
					a.UpdatePropertyCache()
					b.UpdatePropertyCache()
					bonds2 = fm.GetBonds()
			[atom.SetNumRadicalElectrons(0) for atom in atoms2]
			smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
			inchi= AllChem.MolToInchi(fm) 

			#get all resulting combined molecules
			#get InChI output string -> store in the single list..
			new_mols_inchi.append(inchi)

		"""
		elif len(v) == 5:
			#print("make connection")
			edcombo = Chem.EditableMol(main)
			edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.SINGLE)
			ht = []
			for vi in range(len(v)):
				hs = sorted([n.GetIdx() for n in main.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
				if len(hs) > 0:
					ht += hs
			ht = sorted(ht,reverse=True)
			[edcombo.RemoveAtom(t) for t in ht]
			fm = edcombo.GetMol()
			AllChem.Kekulize(fm)
			fm = AllChem.RemoveHs(fm)
			atoms2 = fm.GetAtoms()
			bonds2 = fm.GetBonds()
			
			[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]

			for bi in range(len(bonds2)):
				bond = bonds2[bi]
				a=bond.GetBeginAtom()
				b=bond.GetEndAtom()
				ah=a.GetHybridization()
				bh=b.GetHybridization()
				if ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3:
					bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
					a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
					b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
					a.UpdatePropertyCache()
					b.UpdatePropertyCache()
					bonds2 = fm.GetBonds()
			[atom.SetNumRadicalElectrons(0) for atom in atoms2]
			smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
			inchi= AllChem.MolToInchi(fm) 

			#get all resulting combined molecules
			#get InChI output string -> store in the single list..
			new_mols_inchi.append(inchi)
		"""

	new_mols_inchi = list(set(new_mols_inchi))
	new_mols = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	return new_mols

def Propagate_wO(main,vertx,six_ring=False,five_ring=False,nbfive=0):

	new_mols = []
	new_mols_inchi = []
	#Assume the main mol is aromatic preserving the role of 4N+2 pi electrons
	#Each carbon has a single pi electron
	ring  = main.GetRingInfo()
	ring_members = ring.AtomRings()

	for v in vertx:
		mols = []
		if len(v) == 2: #case1
			if six_ring and not five_ring:		
				sms = ["[CH1][O][CH1]=[CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1]=[CH1][O]"]
			else:
				sms = ["[CH1][O][CH1]=[CH1]","[CH1]=[CH1][O]"]
		
		elif len(v) == 3: #case2
			sms = []
			#if six_ring and not five_ring:
				#sms = ["[CH1][O][CH1]"]
			#elif not six_ring and five_ring:
				#sms = ["[CH1][O]"]
			#else:
				#sms = ["[CH1][O][CH1]","[CH1][O]"]
		
		elif len(v) == 4: #case3
			if six_ring and not five_ring:
				sms = ["[CH1][O]"]
			elif not six_ring and five_ring:
				sms = ["[O]"]
			else:
				sms = ["[CH1][O]","[O]"]

		elif len(v) == 5:
			if six_ring and not five_ring:
				sms = ["[O]"]
			elif not six_ring and five_ring:
				sms = []
			else:
				sms = ["[O]"]
		
		else:
			sms = []

		if len(sms) > 0 and len(v) < 6:
			#print("add rings")
			for typ in range(len(sms)):
				frg = AllChem.MolFromSmiles(sms[typ])
				#find how many of fragment carbons should be SP2 hybridization
				main_atoms = main.GetAtoms()
				frg_atoms = frg.GetAtoms()

				main_Ncarbons = len([atom for atom in main_atoms if atom.GetSymbol() == 'C'])
				frg_Ncarbons = len([atom for atom in frg_atoms if atom.GetSymbol() == 'C'])
				new_Npi = [i for i in range(frg_Ncarbons,-1,-1) if Npi_Aromaticity(i+main_Ncarbons)]

				mcomb = Chem.CombineMols(main,frg)
				mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
				mcomb_atoms = mcomb.GetAtoms()

				mcomb = AllChem.AddHs(mcomb)
				main_idx = mcomb_idx[:-len(frg.GetAtoms())]
				frg_idx = mcomb_idx[-len(frg.GetAtoms()):]

				#atoms to make bonding with frag
				ringmem = [[len(r) for r in ring_members if el in r] for el in v]
				count_five = [rmi.count(5) for rmi in ringmem]
				if any(count_five) <= nbfive:
					edcombo = Chem.EditableMol(mcomb)
					if (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						continue
						#edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						#edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						continue
						#edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						#edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
						continue
						#edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.DOUBLE)
						#edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					else:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)

					ht = []
					for vi in range(len(v)):
						hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
						if len(hs) > 0:
							ht += hs
					ht = sorted(ht,reverse=True)
					[edcombo.RemoveAtom(t) for t in ht]
					fm = edcombo.GetMol()

					atoms2 = fm.GetAtoms()
					bonds2 = fm.GetBonds()
					[atom.SetNumRadicalElectrons(0) for atom in atoms2]
					AllChem.Kekulize(fm)
					fm = AllChem.RemoveHs(fm)

					for ll, atom in enumerate(atoms2):
						bts = []
						for b in atom.GetBonds():
							bts.append(b.GetBondTypeAsDouble())
						valence = sum(bts)+atom.GetTotalNumHs()
						atom.SetNumRadicalElectrons(int(4.5-valence))
					[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]
					#
					#fm = molwith_idx(fm)

					
					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 3 and len(bnn) == 2) or (len(ann) == 2 and len(bnn) == 3):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 2 and len(bnn) == 2):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					[atom.SetNumExplicitHs(2) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]
					[atom.SetNumRadicalElectrons(0) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]

					smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
					inchi= AllChem.MolToInchi(fm) 

					#get all resulting combined molecules
					#get InChI output string -> store in the single list..
					new_mols_inchi.append(inchi)


	new_mols_inchi = list(set(new_mols_inchi))
	new_mols = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	return new_mols


def Find_Vertex_All(m):

	#m should have explicit hydrogens
	m = AllChem.AddHs(m)
	m = AllChem.RemoveHs(m)
	atoms = m.GetAtoms()
	bonds = m.GetBonds()
	

	convex_bond, convex_atom = convex_bond_atom(m)
	
	grow_index = []
	

	for bond_idx in convex_bond:
		bond = bonds[bond_idx]
		t1 = bond.GetBeginAtomIdx()
		t2 = bond.GetEndAtomIdx()
		t1_atom = bond.GetBeginAtom()
		t2_atom = bond.GetEndAtom()
		#Directly bonded
		if t1 in convex_atom and t2 in convex_atom:
			idxs = [t1,t2]
			grow_index.append(idxs)
		#One atom in the middle
		if t1 in convex_atom and t2 not in convex_atom:
			bond2 = t2_atom.GetBonds()
			bb = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			if len(bb) > 0:
				tt1 = bb[0].GetBeginAtomIdx()
				tt2 = bb[0].GetEndAtomIdx()
				if tt1 != t1 and tt1 != t2 and tt1 in convex_atom:
					idxs = [t1,t2,tt1]
					if [tt1,t2,t1] not in grow_index:
						grow_index.append(idxs)
				if tt2 != t1 and tt2 != t2 and tt2 in convex_atom:
					idxs = [t1,t2,tt2]
					if [tt2,t2,t1] not in grow_index:
						grow_index.append(idxs)
		if t1 not in convex_atom and t2 in convex_atom:
			bond2 = t1_atom.GetBonds()
			bb = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			if len(bb) > 0:
				tt1 = bb[0].GetBeginAtomIdx()
				tt2 = bb[0].GetEndAtomIdx()
				if tt1 != t1 and tt1 != t2 and tt1 in convex_atom:
					idxs = [t2,t1,tt1]
					if [tt1,t1,t2] not in grow_index:
						grow_index.append(idxs)
				if tt2 != t1 and tt2 != t2 and tt2 in convex_atom:
					idxs = [t2,t1,tt2]
					if [tt2,t1,t2] not in grow_index:
						grow_index.append(idxs)

		#Two atom in the middle
		if t1 not in convex_atom and t2 not in convex_atom:
			if all([True if n.IsInRing() else False for n in t1_atom.GetNeighbors() ]) and all([True if n.IsInRing() else False for n in t2_atom.GetNeighbors() ]):
				bond1 = t1_atom.GetBonds()
				bond2 = t2_atom.GetBonds()
				bb1 = [b for b in bond1 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
				bb2 = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
				if len(bb1) > 0 and len(bb2) > 0:
					bond1_idx = bb1[0].GetIdx()
					bond2_idx = bb2[0].GetIdx()
					tt1_ = [bb1[0].GetBeginAtomIdx(), bb1[0].GetEndAtomIdx()]
					tt1_.remove(t1)
					tt2_ = [bb2[0].GetBeginAtomIdx(), bb2[0].GetEndAtomIdx()]
					tt2_.remove(t2)
					tmp = tt1_ + tt2_
					tmp2 = [s for s in tmp if s in convex_atom]
					if len(tmp2) == 2:
						idxs = [tmp2[0]] + [t1,t2] + [tmp2[1]]
						grow_index.append(idxs)

					#Four atom in the middle
					if (tt1_[0] not in convex_atom) and (tt2_[0] not in convex_atom):
						bondn1 = atoms[tt1_[0]].GetBonds()
						bondn2 = atoms[tt2_[0]].GetBonds()
						bbb1 = [b for b in bondn1 if b.GetIdx() != bond1_idx and b.GetIdx() in convex_bond]
						bbb2 = [b for b in bondn2 if b.GetIdx() != bond2_idx and b.GetIdx() in convex_bond]
						if len(bbb1) > 0 and len(bbb2) > 0:
							ttt1_ = [bbb1[0].GetBeginAtomIdx(), bbb1[0].GetEndAtomIdx()]
							ttt2_ = [bbb2[0].GetBeginAtomIdx(), bbb2[0].GetEndAtomIdx()]
							ttt1_.remove(tt1_[0])
							ttt2_.remove(tt2_[0])

							if (ttt1_[0] in convex_atom) and (ttt2_[0] in convex_atom):
								idxs = ttt1_ + tt1_ + [t1,t2] + tt2_ + ttt2_
								grow_index.append(idxs)
						
		if t1 not in convex_atom and t2 not in convex_atom:
			bond1 = t1_atom.GetBonds()
			bond2 = t2_atom.GetBonds()
			bb1 = [b for b in bond1 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			bb2 = [b for b in bond2 if b.GetIdx() != bond_idx and b.GetIdx() in convex_bond]
			if len(bb1) > 0 and len(bb2) > 0:
				bond1_idx = bb1[0].GetIdx()
				bond2_idx = bb2[0].GetIdx()
				tt1_ = [bb1[0].GetBeginAtomIdx(), bb1[0].GetEndAtomIdx()]
				tt1_.remove(t1)
				tt2_ = [bb2[0].GetBeginAtomIdx(), bb2[0].GetEndAtomIdx()]
				tt2_.remove(t2)
				tmp = tt1_ + tt2_
				tmp2 = [s for s in tmp if s in convex_atom]
				#Three atom in the middle
				if (tt1_[0] not in convex_atom) and (tt2_[0] in convex_atom):
					bondn1 = atoms[tt1_[0]].GetBonds()
					bbb1 = [b for b in bondn1 if b.GetIdx() != bond1_idx and b.GetIdx() in convex_bond]
					if len(bbb1) > 0:
						ttt1_ = [bbb1[0].GetBeginAtomIdx(), bbb1[0].GetEndAtomIdx()]
						ttt1_.remove(tt1_[0])
						if (ttt1_[0] in convex_atom):
							idxs = ttt1_ + tt1_ + [t1,t2] + tt2_
							grow_index.append(idxs)
				elif (tt1_[0] in convex_atom) and (tt2_[0] not in convex_atom):
					bondn1 = atoms[tt2_[0]].GetBonds()
					bbb1 = [b for b in bondn1 if b.GetIdx() != bond1_idx and b.GetIdx() in convex_bond]
					if len(bbb1) > 0:
						ttt2_ = [bbb1[0].GetBeginAtomIdx(), bbb1[0].GetEndAtomIdx()]
						ttt2_.remove(tt2_[0])
						if (ttt2_[0] in convex_atom):
							idxs = ttt2_ + tt2_ + [t2,t1] + tt1_
							idxs_test = tt1_ + [t1,t2] + tt2_ + ttt2_
							if idxs_test not in grow_index:
								grow_index.append(idxs)

	grow_index = sorted(grow_index,key=lambda x:len(x),reverse=True)

	for i, each in enumerate(grow_index):
		if len(each) == 6:
			test_list = grow_index[i+1:]
			for j, test in enumerate(test_list):
				if each[0] in test or each[-1] in test:
					grow_index.remove(test)
	
	grow_index = [each for each in grow_index \
		          if all([True if atoms[idx].GetSymbol() =='C' else False for idx in each])]
	return grow_index



def Propagate(main,vertx,six_ring=False,five_ring=False,nbfive=0):

	new_mols = []
	new_mols_inchi = []
	#Assume the main mol is aromatic preserving the role of 4N+2 pi electrons
	#Each carbon has a single pi electron
	ring  = main.GetRingInfo()
	ring_members = ring.AtomRings()

	for v in vertx:
		mols = []
		if len(v) == 2: #case1
			if six_ring and not five_ring:		
				sms = ["[CH1][CH1][CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1][CH1][CH1]"]
			else:
				sms = ["[CH1][CH1][CH1][CH1]","[CH1][CH1][CH1]"]
		elif len(v) == 3: #case2
			if six_ring and not five_ring:
				sms = ["[CH1][CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1][CH1]"]
			else:
				sms = ["[CH1][CH1][CH1]","[CH1][CH1]"]
		elif len(v) == 4: #case3
			if six_ring and not five_ring:
				sms = ["[CH1][CH1]"]
			elif not six_ring and five_ring:
				sms = ["[CH1]"]
			else:
				sms = ["[CH1][CH1]","[CH1]"]
		elif len(v) == 5: #case4
			if six_ring and not five_ring:
				sms = ["[CH1]"]
			else:
				sms = []
		elif len(v) == 6: #case5
			sms = []

		if len(sms) > 0 and len(v) < 6:
			for typ in range(len(sms)):
				frg = AllChem.MolFromSmiles(sms[typ])
				#find how many of fragment carbons should be SP2 hybridization
				main_atoms = main.GetAtoms()
				frg_atoms = frg.GetAtoms()

				main_Ncarbons = len([atom for atom in main_atoms if atom.GetSymbol() == 'C'])
				frg_Ncarbons = len([atom for atom in frg_atoms if atom.GetSymbol() == 'C'])
				new_Npi = [i for i in range(frg_Ncarbons,-1,-1) if Npi_Aromaticity(i+main_Ncarbons)]

				mcomb = Chem.CombineMols(main,frg)
				mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]
				mcomb_atoms = mcomb.GetAtoms()

				mcomb = AllChem.AddHs(mcomb)
				main_idx = mcomb_idx[:-len(frg.GetAtoms())]
				frg_idx = mcomb_idx[-len(frg.GetAtoms()):]

				#atoms to make bonding with frag
				ringmem = [[len(r) for r in ring_members if el in r] for el in v]
				count_five = [rmi.count(5) for rmi in ringmem]
				if (len(frg_atoms)+len(v)==6 and any(count_five)>=0) or (len(frg_atoms)+len(v)==5 and any(count_five)<=nbfive):
					edcombo = Chem.EditableMol(mcomb)
					if (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 2:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.DOUBLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 2 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.DOUBLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					elif (mcomb_atoms[v[0]].GetTotalNumHs()) == 1 and (mcomb_atoms[v[-1]].GetTotalNumHs()) == 1:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)
					else:
						edcombo.AddBond(frg_idx[0],v[0],order=Chem.rdchem.BondType.SINGLE)
						edcombo.AddBond(frg_idx[-1],v[-1],order=Chem.rdchem.BondType.SINGLE)

					ht = []
					for vi in range(len(v)):
						hs = sorted([n.GetIdx() for n in mcomb.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
						if len(hs) > 0:
							ht += hs
					ht = sorted(ht,reverse=True)
					[edcombo.RemoveAtom(t) for t in ht]

					fm = edcombo.GetMol()
					AllChem.Kekulize(fm)

					fm = AllChem.RemoveHs(fm)
					atoms2 = fm.GetAtoms()
					bonds2 = fm.GetBonds()
					[atom.SetNumRadicalElectrons(0) for atom in atoms2]

					for ll, atom in enumerate(atoms2):
						bts = []
						for b in atom.GetBonds():
							bts.append(b.GetBondTypeAsDouble())
						valence = sum(bts)+atom.GetTotalNumHs()
						atom.SetNumRadicalElectrons(int(4.5-valence))
					[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]
					
					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 3 and len(bnn) == 2) or (len(ann) == 2 and len(bnn) == 3):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					for bi in range(len(bonds2)):
						bond = bonds2[bi]
						a=bond.GetBeginAtom()
						b=bond.GetEndAtom()
						ah=a.GetHybridization()
						bh=b.GetHybridization()
						ann = [n.GetSymbol() for n in a.GetNeighbors() if n.GetSymbol() == 'C']
						bnn = [n.GetSymbol() for n in b.GetNeighbors() if n.GetSymbol() == 'C']
						a_radial_num = a.GetNumRadicalElectrons()
						b_radial_num = b.GetNumRadicalElectrons()
						if (ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3) and (a_radial_num > 0 and b_radial_num > 0):
							if (len(ann) == 2 and len(bnn) == 2):
								bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
								a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
								a.SetNumRadicalElectrons(a_radial_num-1)
								b.SetNumRadicalElectrons(b_radial_num-1)
								a.UpdatePropertyCache()
								b.UpdatePropertyCache()
								bonds2 = fm.GetBonds()

					[atom.SetNumExplicitHs(2) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]
					[atom.SetNumRadicalElectrons(0) for atom in atoms2 if atom.GetNumRadicalElectrons() == 1]

					smi = AllChem.MolToSmiles(fm)#,kekuleSmiles=True)
					inchi= AllChem.MolToInchi(fm) 

					#get all resulting combined molecules
					#get InChI output string -> store in the single list..
					new_mols_inchi.append(inchi)

		elif len(v) == 5 and len(sms) == 0:
			#print("make connection")
			atoms = main.GetAtoms()
			main2 = AllChem.AddHs(main)
			edcombo = Chem.EditableMol(main2)
			connect = False
			if (atoms[v[0]].GetTotalNumHs()) == 2 and (atoms[v[-1]].GetTotalNumHs()) == 2:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.DOUBLE)
				connect = True
			elif (atoms[v[0]].GetTotalNumHs()) == 1 and (atoms[v[-1]].GetTotalNumHs()) == 1:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.SINGLE)
				connect = True

			if connect:
				ht = []
				for vi in range(len(v)):
					hs = sorted([n.GetIdx() for n in main2.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
					if len(hs) > 0:
						ht += hs
				ht = sorted(ht,reverse=True)
				[edcombo.RemoveAtom(t) for t in ht]
				fm = edcombo.GetMol()
				AllChem.Kekulize(fm)
				fm = AllChem.RemoveHs(fm)
				atoms2 = fm.GetAtoms()
				bonds2 = fm.GetBonds()
				
				[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]

				for bi in range(len(bonds2)):
					bond = bonds2[bi]
					a=bond.GetBeginAtom()
					b=bond.GetEndAtom()
					ah=a.GetHybridization()
					bh=b.GetHybridization()
					if ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3:
						bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
						a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						a.UpdatePropertyCache()
						b.UpdatePropertyCache()
						bonds2 = fm.GetBonds()
				[atom.SetNumRadicalElectrons(0) for atom in atoms2]
				smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
				inchi= AllChem.MolToInchi(fm) 

				#get all resulting combined molecules
				#get InChI output string -> store in the single list..
				new_mols_inchi.append(inchi)

		if len(v) == 6 and len(sms) == 0:
			atoms = main.GetAtoms()
			main2 = AllChem.AddHs(main)
			edcombo = Chem.EditableMol(main2)
			connect = False
			if (atoms[v[0]].GetTotalNumHs()) == 2 and (atoms[v[-1]].GetTotalNumHs()) == 2:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.DOUBLE)
				connect = True
			elif (atoms[v[0]].GetTotalNumHs()) == 1 and (atoms[v[-1]].GetTotalNumHs()) == 1:
				edcombo.AddBond(v[0],v[-1],order=Chem.rdchem.BondType.SINGLE)
				connect = True

			if connect:
				ht = []
				for vi in range(len(v)):
					hs = sorted([n.GetIdx() for n in main2.GetAtoms()[v[vi]].GetNeighbors() if n.GetSymbol()=='H'], reverse=True)
					if len(hs) > 0:
						ht += hs
				ht = sorted(ht,reverse=True)
				[edcombo.RemoveAtom(t) for t in ht]
				fm = edcombo.GetMol()

				AllChem.Kekulize(fm)
				fm = AllChem.RemoveHs(fm)
				atoms2 = fm.GetAtoms()
				bonds2 = fm.GetBonds()
				
				[ atom.SetHybridization(Chem.rdchem.HybridizationType.SP3) for atom in atoms2 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D ]

				for bi in range(len(bonds2)):
					bond = bonds2[bi]
					a=bond.GetBeginAtom()
					b=bond.GetEndAtom()
					ah=a.GetHybridization()
					bh=b.GetHybridization()
					if ah == Chem.rdchem.HybridizationType.SP3 and bh == Chem.rdchem.HybridizationType.SP3:
						bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
						a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
						a.UpdatePropertyCache()
						b.UpdatePropertyCache()
						bonds2 = fm.GetBonds()
				[atom.SetNumRadicalElectrons(0) for atom in atoms2]
				smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
				inchi= AllChem.MolToInchi(fm) 

				#get all resulting combined molecules
				#get InChI output string -> store in the single list..
				new_mols_inchi.append(inchi)

	new_mols_inchi = list(set(new_mols_inchi))
	new_mols = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	return new_mols

def Intramolecular_Bond(mol):
	test_mols = []
	test_info = []

	mol = AllChem.RemoveHs(mol)
	atoms = mol.GetAtoms()
	idx_wH = [atom.GetIdx() for atom in atoms if atom.GetTotalNumHs() > 0]
	nconnect_idxs = []
	h_num_checks = []
	for i in range(len(idx_wH)):
		idx1 = idx_wH[i]
		for j in range(i+1,len(idx_wH)):
			idx2 = idx_wH[j]
			connect_idx = AllChem.GetShortestPath(mol,idx1,idx2)
			Dist3D = AllChem.Get3DDistanceMatrix(mol)
			check_bonded = mol.GetBondBetweenAtoms(idx1,idx2)
			rf = mol.GetRingInfo()
			arf = rf.AtomRings()
			check_ih = [atoms[iid].GetTotalNumHs() for iid in connect_idx]
			check_ir = [atoms[iid].IsInRing() for iid in connect_idx]
			if (Dist3D[idx1,idx2] < 4) and (check_bonded==None) and (not rf.AreAtomsInSameRing(idx1,idx2)):
				mol_copy = copy.deepcopy(mol)
				atoms_copy = mol_copy.GetAtoms()
				#print(idx1,idx2,Dist3D[idx1,idx2],check_bonded)
				#print(atoms_copy[idx1].GetNumExplicitHs(),atoms_copy[idx2].GetNumExplicitHs())

				edcombo = Chem.EditableMol(mol_copy)
				edcombo.AddBond(idx1,idx2,order=Chem.rdchem.BondType.SINGLE)
				back = edcombo.GetMol()

				atoms2 = back.GetAtoms()
				atoms2[idx1].SetNumExplicitHs(0)
				atoms2[idx2].SetNumExplicitHs(0)

				back_H = AllChem.AddHs(back,addCoords=True)
				atoms2_H = back_H.GetAtoms()
				test1 = [n.GetIdx() for n in atoms2_H[idx1].GetNeighbors() if n.GetSymbol() == 'H']
				test2 = [n.GetIdx() for n in atoms2_H[idx2].GetNeighbors() if n.GetSymbol() == 'H']
				em2 = Chem.EditableMol(back_H)
				atomsToRemove = sorted(test1+test2,reverse=True)
				for idd in atomsToRemove:
					em2.RemoveAtom(idd)
				rem_H = em2.GetMol()
				AllChem.Kekulize(rem_H)
				rf2 = rem_H.GetRingInfo()
				arf2 = rf2.AtomRings()
				size_ring = len([c for c in arf2 if len(c) < 5] + [c for c in arf2 if len(c) > 7])
				flag = 0
				for c1, c2 in zip(check_ih[1:-1],check_ir[1:-1]):
					if c1 > 0 and c2 == True:
						flag += 1

				if size_ring == 0 and flag == 0:
					#print(check_ih[1:-1],check_ir[1:-1])
					test_mols.append(rem_H)
					test_info.append("Bonding %s %s" % (idx1,idx2))

	return test_mols, test_info