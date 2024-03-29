import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from .Utility import *
from ase.io import read, write
import numpy as np
import os, sys, glob, subprocess, copy
from itertools import combinations
import itertools
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Geometry import Point3D
from rdkit.Chem.rdMolTransforms import ComputeCentroid

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

def angle_between_plane(n1,n2):
    dot_product = np.dot(n1,n2)
    magnitude_n1 =  np.linalg.norm(n1)
    magnitude_n2 =  np.linalg.norm(n2)
    cos_theta = dot_product / (magnitude_n1 * magnitude_n2)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def best_plane(mol):
	conf = mol.GetConformer()
	c = AllChem.ComputeCentroid(conf)
	centroid = np.array([c.x,c.y,c.z])
	positions = conf.GetPositions()

	_,_, V = np.linalg.svd(positions - centroid)
	normal_vector = V[-1]
	return normal_vector

def mol_plane_and_normalv(mol):
    conf = mol.GetConformer()
    c = AllChem.ComputeCentroid(conf)
    centroid = np.array([c.x,c.y,c.z])
    positions = conf.GetPositions()
    atoms = mol.GetAtoms()
    aromCH_idx = [atom.GetIdx() for atom in atoms if atom.GetIsAromatic() and atom.GetTotalNumHs()>=0]
    p1 = positions[aromCH_idx[0]]
    p2 = positions[aromCH_idx[-1]]

    v1 = p1 - c
    v2 = p2 - c
    normal_vector = np.cross(v1, v2)
    return normal_vector

def break_crosslinking_bond(mol):
	atoms = mol.GetAtoms()
	idxs = [atom.GetIdx() for atom in atoms if atom.IsInRing()]
	pairs = list(combinations(idxs,2))

	frag_pairs = []
	frag_bonds = []
	for pair in pairs:
		path_idx = AllChem.GetShortestPath(mol,pair[0],pair[1])
		check1 = [atoms[pid].IsInRing() for pid in path_idx]
		if (check1[0]==True) and (check1[-1]==True) and list(set(check1[1:-1]))==[False]:
			frag_pairs.append(path_idx)
			b1 = mol.GetBondBetweenAtoms(path_idx[0],path_idx[1])
			b2 = mol.GetBondBetweenAtoms(path_idx[-2],path_idx[-1])
			frag_bonds.append(b1.GetIdx())
			frag_bonds.append(b2.GetIdx())

	if len(frag_bonds)>0:
		mol1_f = Chem.FragmentOnBonds(mol,frag_bonds,addDummies=False)
		mols_f = Chem.GetMolFrags(mol1_f, asMols=True)
		return mols_f
	else:
		return []

def propagate_new(mol,reduce=True,constrained_opt=True,close_ring=[5,6],ring_size=[6],nring_size=[11,12]):
	new_mols = []
	smis = []
	unique = {}
	edges = Find_Vertex_v2(mol)
	frgs_smis = ['C','C=C','C=CC','CC=C','C=CC=C','CC=CC=C','C=CC=CC','C=CC=CC=C',
			     'C=CC=CC=CC','CC=CC=CC=C','C=CC=CC=CC=C','C=CC=CC=CC=CC','CC=CC=CC=CC=C',
			    ]
	frgs = [AllChem.MolFromSmiles(fs) for fs in frgs_smis]
	mol_idx = [atom.GetIdx() for atom in AllChem.RemoveHs(mol).GetAtoms()]
	mol_pos = [mol.GetConformer().GetAtomPosition(ai) for ai, aa in enumerate(mol.GetAtoms())]
	ringinfo = mol.GetRingInfo()
	atomring = ringinfo.AtomRings()
	
	for i, edge in enumerate(edges):
		edge_rinfo1 = []
		for eg in edge:
			check1 = min([len(ar) for ar in atomring if eg in ar])
			#check2 = [True if c in nring_size else False for c in check1 ]			
			edge_rinfo1.append(check1)
		edge_smallest = min(edge_rinfo1)
		
		if len(edge) == 2:
			frgs_screen = [frg for frg in frgs \
				  if (len(frg.GetAtoms())+len(edge) >= min(ring_size)) and (len(frg.GetAtoms())+len(edge) <= max(ring_size)) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest >= nring_size[0]) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest <= nring_size[1])	]
			for frg in frgs_screen:
				new_mol = two_bonds_with_fragment(mol,frg,edge)
				new_mols.append(new_mol)
				smis.append(AllChem.MolToSmiles(AllChem.RemoveHs(new_mol)))
		if len(edge) == 3:
			frgs_screen = [frg for frg in frgs \
				  if (len(frg.GetAtoms())+len(edge) >= min(ring_size)) and (len(frg.GetAtoms())+len(edge) <= max(ring_size)) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest >= nring_size[0]) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest <= nring_size[1])	]
			for frg in frgs_screen:
				new_mol = two_bonds_with_fragment(mol,frg,edge)
				new_mols.append(new_mol)
				smis.append(AllChem.MolToSmiles(AllChem.RemoveHs(new_mol)))
		if len(edge) == 4:
			frgs_screen = [frg for frg in frgs \
				  if (len(frg.GetAtoms())+len(edge) >= min(ring_size)) and (len(frg.GetAtoms())+len(edge) <= max(ring_size)) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest >= nring_size[0]) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest <= nring_size[1])	]
			for frg in frgs_screen:
				new_mol = two_bonds_with_fragment(mol,frg,edge)
				new_mols.append(new_mol)
				smis.append(AllChem.MolToSmiles(AllChem.RemoveHs(new_mol)))
		if len(edge) == 5:
			frgs_screen = [frg for frg in frgs \
				  if (len(frg.GetAtoms())+len(edge) >= min(ring_size)) and (len(frg.GetAtoms())+len(edge) <= max(ring_size)) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest >= nring_size[0]) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest <= nring_size[1])	]
			for frg in frgs_screen:
				new_mol = two_bonds_with_fragment(mol,frg,edge)
				new_mols.append(new_mol)
				smis.append(AllChem.MolToSmiles(AllChem.RemoveHs(new_mol)))
			if len(edge) + edge_smallest >= nring_size[0]:
				new_mol = single_bonds(mol,edge)
				new_mols.append(new_mol)
			smis.append(AllChem.MolToSmiles(AllChem.RemoveHs(new_mol)))
		if len(edge) == 6:
			frgs_screen = [frg for frg in frgs \
				  if (len(frg.GetAtoms())+len(edge) >= min(ring_size)) and (len(frg.GetAtoms())+len(edge) <= max(ring_size)) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest >= nring_size[0]) \
					and (len(frg.GetAtoms())+len(edge)+edge_smallest <= nring_size[1])	]
			for frg in frgs_screen:
				new_mol = two_bonds_with_fragment(mol,frg,edge)
				new_mols.append(new_mol)
				smis.append(AllChem.MolToSmiles(AllChem.RemoveHs(new_mol)))
			if len(edge) + edge_smallest >= nring_size[0]:
				new_mol = single_bonds(mol,edge)
				new_mols.append(new_mol)
			smis.append(AllChem.MolToSmiles(AllChem.RemoveHs(new_mol)))

	for j, new_mol in enumerate(new_mols):
		flag = AllChem.RemoveHs(new_mol).HasSubstructMatch(AllChem.RemoveHs(mol))
		atom_idx = AllChem.RemoveHs(new_mol).GetSubstructMatch(AllChem.RemoveHs(mol))
		#print(flag, atom_idx)
		#Draw.MolToImageFile(new_mol,'test_%d.png' % (j+1),highlightAtoms=mol_idx)
	if reduce:
		for new_mol in new_mols:
			unique[AllChem.MolToSmiles(new_mol)] = new_mol
		new_mols = list(unique.values())

	#new_mols = [AllChem.AddHs(new_mol,addCoords=True) for new_mol in new_mols]
	new_mols = [AllChem.RemoveHs(new_mol) for new_mol in new_mols]
	return new_mols

def single_bonds(mol,edge):
	#Able to deal with cove and fjord region

	mol = AllChem.RemoveHs(mol)
	combH = AllChem.AddHs(mol,addCoords=True)
	atoms_combH = combH.GetAtoms()
	connect = False
	edge1 = edge[0]
	edge2 = edge[-1]
	hidxs1_mol = [n.GetIdx() for n in atoms_combH[edge1].GetNeighbors() if n.GetSymbol() == 'H']
	hidxs2_mol = [n.GetIdx() for n in atoms_combH[edge2].GetNeighbors() if n.GetSymbol() == 'H']

	h_remove = []
	EAddFrg = AllChem.EditableMol(combH)

	if (len(hidxs1_mol) >= 2) and (len(hidxs2_mol) >= 2):
		EAddFrg.AddBond(edge1,edge2,order=Chem.rdchem.BondType.DOUBLE)
		for hi in range(2):
			h_remove.append(hidxs1_mol[-1])
			hidxs1_mol.remove(hidxs1_mol[-1])
			h_remove.append(hidxs2_mol[-1])
			hidxs2_mol.remove(hidxs2_mol[-1])
		connect = True
	else:
		EAddFrg.AddBond(edge1,edge2,order=Chem.rdchem.BondType.SINGLE)
		for hi in range(1):
			h_remove.append(hidxs1_mol[-1])
			hidxs1_mol.remove(hidxs1_mol[-1])
			h_remove.append(hidxs2_mol[-1])
			hidxs2_mol.remove(hidxs2_mol[-1])
		connect = True

	h_remove = sorted(h_remove,reverse=True)

	for h_idx in h_remove:
		EAddFrg.RemoveAtom(h_idx)
	mol_new = EAddFrg.GetMol()
	#AllChem.EmbedMolecule(mol_new)
	AllChem.MMFFOptimizeMolecule(mol_new, mmffVariant='MMFF94s')
	mol_new = AllChem.RemoveHs(mol_new)
	return mol_new

def two_bonds_with_fragment(mol,frg,edge):
	#Able to deal with cove, K-region, L-region, bay region with additional fragments

	mol = AllChem.RemoveHs(mol)
	mol_idx = [atom.GetIdx() for atom in AllChem.RemoveHs(mol).GetAtoms()]
	mol_edge_ave_positions = np.mean(np.array([mol.GetConformer().GetAtomPosition(eg) for eg in edge]),axis=0)
	mol_center = np.array(ComputeCentroid(mol.GetConformer()))
	v = mol_edge_ave_positions - mol_center
	final_point = 2*v + mol_center

	frg = AllChem.AddHs(frg)
	AllChem.EmbedMolecule(frg, useRandomCoords=False, useBasicKnowledge=False)
	AllChem.MMFFOptimizeMolecule(frg, mmffVariant='MMFF94s',nonBondedThresh=1000)
	frg = AllChem.RemoveHs(frg)
	frg_center = np.array(ComputeCentroid(frg.GetConformer()))
	dis = frg_center - final_point
	frg_pos = [frg.GetConformer().GetAtomPosition(ai) for ai, aa in enumerate(frg.GetAtoms())]
	frg_new = [np.array(p)-dis for p in frg_pos]
	conf = frg.GetConformer()
	for i, ap in enumerate(frg_new):
		x,y,z = ap
		conf.SetAtomPosition(i,Point3D(x,y,z))
	frg_center = np.array(ComputeCentroid(frg.GetConformer()))

	atoms_mol = mol.GetAtoms()
	atoms_frg = frg.GetAtoms()

	frg_idx =  [int(ids) for ids in np.arange(len(atoms_frg)) + len(atoms_mol)]
	comb = Chem.CombineMols(mol,frg)
	combH = AllChem.AddHs(comb,addCoords=True)
	atoms_combH = combH.GetAtoms()

	edge1_hyb = atoms_combH[edge[0]].GetHybridization()
	edge2_hyb = atoms_combH[edge[-1]].GetHybridization()
	frg1_hyb = atoms_combH[frg_idx[0]].GetHybridization()
	frg2_hyb = atoms_combH[frg_idx[-1]].GetHybridization()

	hidxs1_mol = [n.GetIdx() for n in atoms_combH[edge[0]].GetNeighbors() if n.GetSymbol() == 'H']
	hidxs2_mol = [n.GetIdx() for n in atoms_combH[edge[-1]].GetNeighbors() if n.GetSymbol() == 'H']
	hidxs1_frg = [n.GetIdx() for n in atoms_combH[frg_idx[0]].GetNeighbors() if n.GetSymbol() == 'H']
	hidxs2_frg = [n.GetIdx() for n in atoms_combH[frg_idx[-1]].GetNeighbors() if n.GetSymbol() == 'H']

	if len(atoms_frg) == 1:
		hidxs1_frg = hidxs2_frg

	h_remove = []
	EAddFrg = AllChem.EditableMol(combH)
	#print(edge1_hyb, edge2_hyb, frg1_hyb, frg2_hyb)
	if ((edge1_hyb == HybridizationType.SP3) and (frg1_hyb == HybridizationType.SP3) and (frg_idx[0]!=frg_idx[-1]) and (len(hidxs1_mol)>=2) and (len(hidxs1_frg)>=2)):
		EAddFrg.AddBond(edge[0],frg_idx[0],order=Chem.rdchem.BondType.DOUBLE)
		for hi in range(2):
			h_remove.append(hidxs1_mol[-1])
			hidxs1_mol.remove(hidxs1_mol[-1])
			h_remove.append(hidxs1_frg[-1])
			hidxs1_frg.remove(hidxs1_frg[-1])
		
	else:
		EAddFrg.AddBond(edge[0],frg_idx[0],order=Chem.rdchem.BondType.SINGLE)
		for hi in range(1):
			h_remove.append(hidxs1_mol[-1])
			hidxs1_mol.remove(hidxs1_mol[-1])
			h_remove.append(hidxs1_frg[-1])
			hidxs1_frg.remove(hidxs1_frg[-1])

	if ((edge2_hyb == HybridizationType.SP3) and (frg2_hyb == HybridizationType.SP3) and (frg_idx[0]!=frg_idx[-1]) and (len(hidxs2_mol)>=2) and (len(hidxs2_frg)>=2)):
		EAddFrg.AddBond(edge[-1],frg_idx[-1],order=Chem.rdchem.BondType.DOUBLE)
		for hi in range(2):
			h_remove.append(hidxs2_mol[-1])
			hidxs2_mol.remove(hidxs2_mol[-1])
			h_remove.append(hidxs2_frg[-1])
			hidxs2_frg.remove(hidxs2_frg[-1])
	
	else:
		EAddFrg.AddBond(edge[-1],frg_idx[-1],order=Chem.rdchem.BondType.SINGLE)
		for hi in range(1):
			h_remove.append(hidxs2_mol[-1])
			hidxs2_mol.remove(hidxs2_mol[-1])
			h_remove.append(hidxs2_frg[-1])
			hidxs2_frg.remove(hidxs2_frg[-1])
	h_remove = sorted(h_remove,reverse=True)

	for h_idx in h_remove:
		EAddFrg.RemoveAtom(h_idx)
	mol_new = EAddFrg.GetMol()	
	mmffps = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_new)
	ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_new,mmffps)
	for atidx in mol_idx:
		ff.MMFFAddPositionConstraint(atidx,0.05,200)
	ff.Minimize()
	
	mol_new = AllChem.RemoveHs(mol_new)
	return mol_new

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
				sms = ["[CH1]","[CH2]"]
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

def Intramolecular_Bond(mol,restrictions=True,constrained_opt=True):
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

			sym1 = atoms[idx1].GetSymbol()
			sym2 = atoms[idx2].GetSymbol()
			
			pair_exclude = ['OO','NN','SS']
			flag_sym = ((sym1+sym2 in pair_exclude) or (sym2+sym1 in pair_exclude))
			#print(Dist3D[idx1,idx2])
			if (not flag_sym) and (Dist3D[idx1,idx2] < 4) and (check_bonded==None) and (not rf.AreAtomsInSameRing(idx1,idx2)):
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
				try:
					rem = AllChem.RemoveHs(rem_H)
					rem_H = AllChem.AddHs(rem,addCoords=True)
					AllChem.Kekulize(rem_H)

				except:
					continue
				
				if constrained_opt:
					pos_info = [[ai,rem_H.GetConformer().GetAtomPosition(ai)] for ai, aa in enumerate(rem_H.GetAtoms())]
					pos = [[p.x,p.y,p.z] for atom_idx, p in pos_info]
					dis_th = 8.0
					check1 = [vi for vi,dis_v in enumerate(np.array(pos) - pos[idx1]) if np.linalg.norm(dis_v) > dis_th]
					check2 = [vi for vi,dis_v in enumerate(np.array(pos) - pos[idx2]) if np.linalg.norm(dis_v) > dis_th]
					constrain_idxs = sorted(set(check1+check2))
					mmffps = rdForceFieldHelpers.MMFFGetMoleculeProperties(rem_H)
					ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(rem_H,mmffps)
					for atidx in constrain_idxs:
						ff.MMFFAddPositionConstraint(atidx,0.05,200)
					ff.Minimize()
					
				rf2 = rem_H.GetRingInfo()
				arf2 = rf2.AtomRings()
				flag = 0
				size_ring = len([c for c in arf2 if len(c) < 5])
				
				if restrictions:
					size_ring = len([c for c in arf2 if len(c) < 5] + [c for c in arf2 if len(c) > 7])
					for c1, c2 in zip(check_ih[1:-1],check_ir[1:-1]):
						if c1 > 0 and c2 == True:
							flag += 1
				if size_ring == 0 and flag == 0:
					test_mols.append(rem_H)
					test_info.append("Bonding %s %s" % (idx1,idx2))
	return test_mols, test_info


def Intramolecular_Bond_v2(mol,bdist=4,restrictions=True):
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

			sym1 = atoms[idx1].GetSymbol()
			sym2 = atoms[idx2].GetSymbol()
			
			pair_exclude = ['OO','NN','SS']
			flag_sym = ((sym1+sym2 in pair_exclude) or (sym2+sym1 in pair_exclude))
			#print(Dist3D[idx1,idx2])
			if (not flag_sym) and (Dist3D[idx1,idx2] <= bdist) and (check_bonded==None) and (not rf.AreAtomsInSameRing(idx1,idx2)):
				mol_copy = copy.deepcopy(mol)
				atoms_copy = mol_copy.GetAtoms()

				edcombo = Chem.EditableMol(mol_copy)
				edcombo.AddBond(idx1,idx2,order=Chem.rdchem.BondType.SINGLE)
				back = edcombo.GetMol()

				atoms2 = back.GetAtoms()
				atoms2[idx1].SetNumExplicitHs(0)
				atoms2[idx2].SetNumExplicitHs(0)

				back_H = AllChem.AddHs(back,addCoords=True)
				atoms2_H = back_H.GetAtoms()
				hidxs1 = sorted([n.GetIdx() for n in atoms2_H[idx1].GetNeighbors() if n.GetSymbol() == 'H'],reverse=True)
				hidxs2 = sorted([n.GetIdx() for n in atoms2_H[idx2].GetNeighbors() if n.GetSymbol() == 'H'],reverse=True)
				em2 = Chem.EditableMol(back_H)
				atomsToRemove = sorted([hidxs1[0],hidxs2[0]],reverse=True)
				for idd in atomsToRemove:
					em2.RemoveAtom(idd)
				rem_H = em2.GetMol()
				try:
					rem = AllChem.RemoveHs(rem_H)
					rem_H = AllChem.AddHs(rem,addCoords=True)
					AllChem.Kekulize(rem_H)

				except:
					continue

				pos_info = [[ai,rem_H.GetConformer().GetAtomPosition(ai)] for ai, aa in enumerate(rem_H.GetAtoms())]
				pos = [[p.x,p.y,p.z] for atom_idx, p in pos_info]
				dis_th = 8.0
				check1 = [vi for vi,dis_v in enumerate(np.array(pos) - pos[idx1]) if np.linalg.norm(dis_v) > dis_th]
				check2 = [vi for vi,dis_v in enumerate(np.array(pos) - pos[idx2]) if np.linalg.norm(dis_v) > dis_th]
				constrain_idxs = sorted(set(check1+check2))
				mmffps = rdForceFieldHelpers.MMFFGetMoleculeProperties(rem_H)
				ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(rem_H,mmffps)
				for atidx in constrain_idxs:
					ff.MMFFAddPositionConstraint(atidx,0.05,200)
				ff.Minimize()
				
				rf2 = rem_H.GetRingInfo()
				arf2 = rf2.AtomRings()
				flag = 0
				size_ring = len([c for c in arf2 if len(c) < 5])
				
				if restrictions:
					size_ring = len([c for c in arf2 if len(c) < 5] + [c for c in arf2 if len(c) > 7])
					for c1, c2 in zip(check_ih[1:-1],check_ir[1:-1]):
						if c1 > 0 and c2 == True:
							flag += 1
				if size_ring == 0 and flag == 0:
					test_mols.append(rem_H)
					test_info.append("Bonding %s %s" % (idx1,idx2))
	return test_mols, test_info
