import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase.io import read, write
import numpy as np
import os, sys, glob, subprocess
from itertools import combinations
import itertools

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
	
	m = AllChem.RemoveHs(m,updateExplicitCount=True)
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
	ExH = [atom.GetNumExplicitHs() for atom in atoms]
	for bond in convex_bond:
		a = bond.GetBeginAtom()
		b = bond.GetEndAtom()
		aidx = bond.GetBeginAtomIdx()
		bidx = bond.GetEndAtomIdx()
		if ExH[aidx] > 0:
			convex_atom.append(a)
			convex_atom_idx.append(aidx)
		if ExH[bidx] > 0:
			convex_atom.append(b)
			convex_atom_idx.append(bidx)
		convex_atom_idx = list(set(convex_atom_idx))

	return convex_bond_idx, convex_atom_idx

def Find_Vertex_v2(m):

	#m should have explicit hydrogens
	m = AllChem.AddHs(m)
	m = AllChem.RemoveHs(m,updateExplicitCount=True)
	convex_bond, convex_atom = convex_bond_atom(m)
	#print(convex_bond, convex_atom)

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
				ttt1_ = [bbb1[0].GetBeginAtomIdx(), bbb1[0].GetEndAtomIdx()]
				ttt1_.remove(tt1_[0])
				if (ttt1_[0] in convex_atom):
					idxs = ttt1_ + tt1_ + [t1,t2] + tt2_
					grow_index.append(idxs)
			elif (tt1_[0] in convex_atom) and (tt2_[0] not in convex_atom):
				bondn1 = atoms[tt2_[0]].GetBonds()
				bbb1 = [b for b in bondn1 if b.GetIdx() != bond1_idx and b.GetIdx() in convex_bond]
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
					
	return grow_index


def Propagate_v2(main,vertx):

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
			sms = ["[CH1][CH1][CH1][CH1]","[CH1][CH1][CH1]"]
		elif len(v) == 3: #case2
			sms = ["[CH1][CH1][CH1]","[CH1][CH1]"]
		elif len(v) == 4: #case3
			sms = ["[CH1][CH1]","[CH1]"]
		elif len(v) == 5: #case4
			sms = ["[CH1]"]
		elif len(v) == 6: #case5
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
				if any(count_five) < 1:
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

					#fm = AllChem.AddHs(fm)
					AllChem.Compute2DCoords(fm)
					Chem.Draw.MolToFile(fm,'./test.png',size=(800,800),kekulize=True)
					#subprocess.call('imgcat ./test.png', shell=True)

					fm = AllChem.RemoveHs(fm)
					atoms2 = fm.GetAtoms()
					bonds2 = fm.GetBonds()
					[atom.SetNumRadicalElectrons(0) for atom in atoms2]

					for ll, atom in enumerate(atoms2):
						bts = []
						for b in atom.GetBonds():
							bts.append(b.GetBondTypeAsDouble())
						valence = sum(bts)+atom.GetTotalNumHs()
						#print(ll,bts,sum(bts),atom.GetTotalNumHs())
						#print(4-valence)
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
		if len(v) == 6:
			#print("make connection",v)
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

def Propagate_Aliphatic(main,vertx):

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
			sms = ["[CH2][CH2][CH2][CH2]","[CH2][CH2][CH2]"]
		elif len(v) == 3: #case2
			sms = ["[CH2][CH2][CH2]","[CH2][CH2]"]
		elif len(v) == 4: #case3
			sms = ["[CH2][CH2]","[CH2]"]
		elif len(v) == 5: #case4
			sms = ["[CH2]"]
		#elif len(v) == 6: #case5
		#	sms = []

		flag = (main.GetAtoms()[v[0]].GetTotalNumHs() == 1 and main.GetAtoms()[v[-1]].GetTotalNumHs() == 1)

		if len(sms) > 0 and len(v) < 6 and flag:
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
				if any(count_five) < 2:
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
						#print(ll,bts,sum(bts),atom.GetTotalNumHs())
						#print(4-valence)
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
					[atom.SetNumRadicalElectrons(0) for atom in atoms2 if atom.GetNumRadicalElectrons() >= 1]

					#fm = AllChem.AddHs(fm)
					#AllChem.Compute2DCoords(fm)
					#Chem.Draw.MolToFile(fm,'./test.png',size=(800,800),kekulize=True)
					#subprocess.call('imgcat ./test.png', shell=True)

					smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
					inchi= AllChem.MolToInchi(fm) 

					#get all resulting combined molecules
					#get InChI output string -> store in the single list..
					new_mols_inchi.append(inchi)

	new_mols_inchi = list(set(new_mols_inchi))
	new_mols = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	return new_mols






def Find_Vertex(m):
	atoms = m.GetAtoms()
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

	outbond = []
	outbond_idx = []
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
			outbond.append(bond)
			outbond_idx.append(idx)

	#outbond is the list of bond to add ring components
	vertx = {}
	vertx["case1"] = []
	vertx["case2"] = []
	vertx["case3"] = []

	for i in outbond_idx:
		bond = m.GetBondWithIdx(i)
		a = bond.GetBeginAtom()
		b = bond.GetEndAtom()
		ann = a.GetNeighbors()
		bnn = b.GetNeighbors()

		#Case 1
		if len(ann) == 2 and len(bnn) == 2:
			edge_atom = [a.GetIdx(), b.GetIdx()]
			test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
			vertx["case1"].append(edge_atom)

		#Case 2-1
		elif (len(ann) == 2 and len(bnn) == 3):
			b_idx = b.GetIdx()
			bnn_idx = [bn.GetIdx() for bn in bnn]
			bnn_bonded = [bn_idx for bn_idx in bnn_idx if len(m.GetAtoms()[bn_idx].GetNeighbors()) == 2]
			bnn_bonded2 = [bn_idx for bn_idx in bnn_idx if len(m.GetAtoms()[bn_idx].GetNeighbors()) == 1]

			if a.GetIdx() in bnn_bonded:
				bnn_bonded.remove(a.GetIdx())
			if b.GetIdx() in bnn_bonded:
				bnn_bonded.remove(b.GetIdx())

			if a.GetIdx() in bnn_bonded2:
				bnn_bonded2.remove(a.GetIdx())
			if b.GetIdx() in bnn_bonded2:
				bnn_bonded2.remove(b.GetIdx())
		
			#2-3-2
			if len(bnn_bonded) == 1:
				edge_atom = [a.GetIdx(),b.GetIdx(),bnn_bonded[0]]
				test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
				vertx["case2"].append(edge_atom)

			#2-3-1
			if len(bnn_bonded2) == 1:
				edge_atom = [a.GetIdx(),b.GetIdx(),bnn_bonded2[0]]
				test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
				vertx["case2"].append(edge_atom)

		#Case 3
		elif len(ann) == 3 and len(bnn) == 3:
			a_idx = a.GetIdx()
			b_idx = b.GetIdx()
			ann_idx = [an.GetIdx() for an in ann]
			bnn_idx = [bn.GetIdx() for bn in bnn]

			ann_idx.remove(b_idx)
			bnn_idx.remove(a_idx)

			ann_bonded_on = [an_idx for an_idx in ann_idx if len(m.GetAtoms()[an_idx].GetNeighbors()) == 1]
			bnn_bonded_on = [bn_idx for bn_idx in bnn_idx if len(m.GetAtoms()[bn_idx].GetNeighbors()) == 1]
			ann_bonded_tn = [an_idx for an_idx in ann_idx if len(m.GetAtoms()[an_idx].GetNeighbors()) == 2]
			bnn_bonded_tn = [bn_idx for bn_idx in bnn_idx if len(m.GetAtoms()[bn_idx].GetNeighbors()) == 2]

			#2-3-3-1
			if len(ann_bonded_tn) == 1 and len(bnn_bonded_on) == 1:
				edge_atom = [ann_bonded_tn[0],a.GetIdx(),b.GetIdx(),bnn_bonded_on[0]]
				test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
				vertx["case3"].append(edge_atom)
			#1-3-3-2
			if len(ann_bonded_tn) == 1 and len(bnn_bonded_tn) == 1 and len(ann_bonded_on) == 1 and len(bnn_bonded_on) == 0:
				edge_atom = [ann_bonded_on[0],a.GetIdx(),b.GetIdx(), bnn_bonded_tn[0]]
				test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
				vertx["case3"].append(edge_atom)
			#2-3-3-1
			if len(ann_bonded_tn) == 1 and len(bnn_bonded_tn) == 1 and len(ann_bonded_on) == 0 and len(bnn_bonded_on) == 1:
				edge_atom = [ann_bonded_tn[0],a.GetIdx(),b.GetIdx(),bnn_bonded_on[0]]
				test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
				vertx["case3"].append(edge_atom)
			#2-3-3-2
			if len(ann_bonded_tn) == 1 and len(bnn_bonded_tn) == 1 and len(ann_bonded_on) == 0 and len(bnn_bonded_on) == 0:
				edge_atom = [ann_bonded_tn[0],a.GetIdx(),b.GetIdx(),bnn_bonded_tn[0]]
				test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
				vertx["case3"].append(edge_atom)
			#1-3-3-1
			if len(ann_bonded_tn) == 1 and len(bnn_bonded_tn) == 1 and len(ann_bonded_on) == 1 and len(bnn_bonded_on) == 1:
				edge_atom = [ann_bonded_on[0],a.GetIdx(),b.GetIdx(),bnn_bonded_on[0]]
				test = [[len(r) for r in ring_members if v in r] for v in edge_atom]
				vertx["case3"].append(edge_atom)


		#Case 4
		#elif len(ann) == 3 and len(bnn) == 3:
	return vertx




def Propagate(main,vertx):

	new_mols = []
	new_mols_inchi = []

	#imH = [atom.GetNumImplicitHs() for atom in main.GetAtoms()]
	#[atom.SetNoImplicit(True) for atom in main.GetAtoms()]
	#[atom.SetNumExplicitHs(imH[a]) for a, atom in enumerate(main.GetAtoms())]

	#Assume the main mol is aromatic preserving the role of 4N+2 pi electrons
	#Each carbon has a single pi electron
	ring  = main.GetRingInfo()
	ring_members = ring.AtomRings()

	#main_idx = [atom.GetIdx() for atom in main.GetAtoms()]
	for key, values in vertx.items():
		mols = []
		if key == "case1":		
			sms = ["[CH1][CH1][CH1][CH1]","[CH1][CH1][CH1]"]
		elif key == "case2":
			sms = ["[CH1][CH1][CH1]","[CH1][CH1]"]
		elif key == "case3":
			sms = ["[CH1][CH1]","[CH1]"]

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

			mcomb = AllChem.AddHs(mcomb)
			main_idx = mcomb_idx[:-len(frg.GetAtoms())]
			frg_idx = mcomb_idx[-len(frg.GetAtoms()):]

			for i in range(len(values)):
				#atoms to make bonding with frag
				v = values[i]
				ringmem = [[len(r) for r in ring_members if el in r] for el in v]
				count_five = [rmi.count(5) for rmi in ringmem]
				if any(count_five) < 1:
					edcombo = Chem.EditableMol(mcomb)
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
					"""
					print( [atom.GetHybridization() for atom in atoms2]) 
					print( [atom.GetNumExplicitHs() for atom in atoms2]) 
					print( [atom.GetNumImplicitHs() for atom in atoms2]) 
					print( [atom.GetExplicitValence() for atom in atoms2])
					print( [atom.GetIsAromatic() for atom in atoms2]) 
					print( [bond.GetBondType() for bond in bonds2] )
					"""
					smi = AllChem.MolToSmiles(fm,kekuleSmiles=True)
					inchi= AllChem.MolToInchi(fm) 
					print(inchi)					

					#get all resulting combined molecules
					#get InChI output string -> store in the single list..
					new_mols_inchi.append(inchi)

	new_mols_inchi = list(set(new_mols_inchi))
	new_mols = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	return new_mols

def Propagate_old_ref(main,vertx):

	#Assume the main mol is aromatic preserving the role of 4N+2 pi electrons
	#Each carbon has a single pi electron
	atoms = main.GetAtoms()
	Npi = len([atom for atom in atoms if atom.GetSymbol()=='C'])

	ring  = main.GetRingInfo()
	ring_members = ring.AtomRings()

	main_idx = [atom.GetIdx() for atom in main.GetAtoms()]

	new_mols = []
	new_mols_inchi = []
	for key, values in vertx.items():
		mols = []
		if key == "case1":		
			sms = ["[CH1][CH1][CH1][CH1]", "[CH1][CH1][CH1]"]
		elif key == "case2":
			sms = ["[CH1][CH1][CH1]", "[CH1][CH1]"]
		elif key == "case3":
			sms = ["[CH1][CH1]", "[CH1]"]

		for typ in range(len(sms)):
			frg = AllChem.MolFromSmiles(sms[typ])
			####
			#find how many of fragment carbons should be SP2 hybridization
			main_atoms = main.GetAtoms()
			frg_atoms = frg.GetAtoms()
			main_Ncarbons = len([atom for atom in main_atoms if atom.GetSymbol() == 'C'])
			frg_Ncarbons = len([atom for atom in frg_atoms if atom.GetSymbol() == 'C'])
			new_Npi = [i for i in range(frg_Ncarbons,-1,-1) if Npi_Aromaticity(i+main_Ncarbons)]

			mcomb = Chem.CombineMols(main,frg)
			mcomb_idx = [atom.GetIdx() for atom in mcomb.GetAtoms()]

			mcomb = AllChem.AddHs(mcomb)
			main_idx = mcomb_idx[:-len(frg.GetAtoms())]
			frg_idx = mcomb_idx[-len(frg.GetAtoms()):]

			for i in range(len(values)):
				v = values[i]
				ringmem = [[len(r) for r in ring_members if el in r] for el in v]
				count_five = [rmi.count(5) for rmi in ringmem]
				if any(count_five) < 1:
					edcombo = Chem.EditableMol(mcomb)
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
					[atoms2[f].SetHybridization(rdkit.Chem.rdchem.HybridizationType.SP3) for f in main_idx]
					bonds2 = fm.GetBonds()
					#combination of fragment conversion
					if len(new_Npi) > 0:
						for comb in new_Npi:
							sp2_hyb_idx = list(combinations(frg_idx,comb))
							for spidx in sp2_hyb_idx:
								addhs_idx = [f for f in frg_idx if f not in spidx]
								#[atoms2[sp2].SetNumExplicitHs(1) for sp2 in addhs_idx]
								#[atoms2[sp2].SetNumExplicitHs(1) for sp2 in addhs_idx]

								#get all resulting combined molecules
								#get InChI output string -> store in the single list..
								fm = AllChem.RemoveHs(fm)
								atoms2 = fm.GetAtoms()
								ring2  = fm.GetRingInfo()
								ring_members2 = ring2.AtomRings()
								six_mem_a = [each for each in ring_members2 if len(each) == 6]
								ring_bonds = ring2.BondRings()
								six_mem_b = [each for each in ring_bonds if len(each) == 6]

								[atom.SetNumRadicalElectrons(0) for atom in atoms2]
								atoms2 = fm.GetAtoms()
								[atoms2[sp2].SetHybridization(rdkit.Chem.rdchem.HybridizationType.SP2) for sp2 in sp2_hyb_idx[0]]
								bonds2 = fm.GetBonds()

								ring  = fm.GetRingInfo()
								ring_members = ring.AtomRings()
								#Set aromatic True for atom if 6 or 5 members are all sp2
								for rn in ring_members:
									hyd = [ atoms2[idx].GetHybridization() for idx in rn ]
									setaromatic = all(hyd[0] == elem for elem in hyd)	
									if setaromatic and len(rn) > 4 and len(rn) < 10 :
										[ atoms2[idx].SetIsAromatic(True) for idx in rn ]
										[ bond.SetIsAromatic(True) for bond in bonds2 if bond.GetBeginAtomIdx() in rn and bond.GetEndAtomIdx() in rn ]

								#print( [atom.GetHybridization() for atom in atoms2]) 
								#print( [atom.GetNumExplicitHs() for atom in atoms2]) 
								#print( [atom.GetNumImplicitHs() for atom in atoms2]) 
								#print( [atom.GetExplicitValence() for atom in atoms2])
								#print( [atom.GetIsAromatic() for atom in atoms2]) 
								#print( [bond.GetBondType() for bond in bonds2] )
								#new_mols.append(fm)

								try:
									inchi= AllChem.MolToInchi(fm) 
									new_mols_inchi.append(inchi)
								except:
									continue
									"""
									highlight_atoms = v + frg_idx
									highlight_bonds = []
									hb = [[bond.GetIdx() for bond in atoms2[i].GetBonds()] for i in highlight_atoms]
									[[highlight_bonds.append(b) for b in bb] for bb in hb]
									highlight_bonds = [bond.GetIdx() for bond in fm.GetBonds() if bond.GetBeginAtomIdx() in highlight_atoms and  bond.GetEndAtomIdx() in highlight_atoms ]

									"""

					"""
					for rn in ring_members:
						hyd = [ atoms2[idx].GetHybridization() for idx in rn ]
						setaromatic_flag = all([elem == Chem.rdchem.HybridizationType.SP2 for elem in hyd])
						print(rn, setaromatic_flag, hyd)
						if len(rn) >= 5 and setaromatic_flag:
							[ atoms2[idx].SetIsAromatic(True) for idx in rn ]
							bonds2 = fm.GetBonds()
							for bond in bonds2:
								a = bond.GetBeginAtom()
								b = bond.GetEndAtom()
								if a.GetIsAromatic() and b.GetIsAromatic():
									bond.SetIsAromatic(True)
					bonds2 = fm.GetBonds()
					print("Aromatic bond", [bond.GetIsAromatic() for bond in bonds2]) 
					print("Explicit", [atom.GetNumExplicitHs() for atom in fm.GetAtoms()]) 
					print("Implicit", [atom.GetNumImplicitHs() for atom in fm.GetAtoms()]) 
					print("Valence", [atom.GetTotalValence() for atom in fm.GetAtoms()])

					##
					highlight_atoms = v + frg_idx
					highlight_bonds = []
					hb = [[bond.GetIdx() for bond in atoms2[i].GetBonds()] for i in highlight_atoms]
					[[highlight_bonds.append(b) for b in bb] for bb in hb]
					highlight_bonds = [bond.GetIdx() for bond in fm.GetBonds() if bond.GetBeginAtomIdx() in highlight_atoms and  bond.GetEndAtomIdx() in highlight_atoms ]

					#fm = molwith_idx(fm)
					#fm = AllChem.AddHs(fm)
					AllChem.Compute2DCoords(fm)
					Chem.Draw.MolToFile(fm,'./check.png',size=(800,800),kekulize=True,highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds)
					subprocess.call('imgcat ./check.png', shell=True)
					##
					"""

	new_mols_inchi = list(set(new_mols_inchi))
	new_mols = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	for i in range(len(new_mols)):
		m = new_mols[i]
		atoms = m.GetAtoms()
		bonds = m.GetBonds()
		for bond in bonds:
			bondtype = bond.GetBondType()
			a = bond.GetBeginAtom()
			b = bond.GetEndAtom()
			if bondtype == Chem.rdchem.BondType.SINGLE and a.GetHybridization() == Chem.rdchem.HybridizationType.SP3 and b.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
				bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
				a.SetHybridization(Chem.rdchem.HybridizationType.SP2)
				b.SetHybridization(Chem.rdchem.HybridizationType.SP2)
				
				inchi2 = AllChem.MolToInchi(m)
				new_mols_inchi.append(inchi2)

				
	new_mols_inchi = list(set(new_mols_inchi))
	#print(new_mols_inchi)
	new_mols2 = [AllChem.MolFromInchi(inchi) for inchi in new_mols_inchi]

	return new_mols2
