from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from .Hydrocarbons_Builder import convex_bond_atom
import random
import os, sys, subprocess, glob
from copy import deepcopy

def periRingsize_convert(mol, rsizefrom, rsizeto):

	mols = []
	mol = AllChem.AddHs(mol)
	mol = AllChem.RemoveHs(mol)
	Chem.Kekulize(mol)
	#convex_bond_idx, convex_atom_idx = convex_bond_atom(mol)
	if rsizefrom == 6 and rsizeto == 5:
		atoms = mol.GetAtoms()
		bonds = mol.GetBonds()

		ring = mol.GetRingInfo()
		aring = ring.AtomRings()
		bring = ring.BondRings()
		bring_6 = [br for br in bring if len(br)]
		for bi, br in enumerate(bring_6):
			bridge_bond = []
			bridgehead_bond = []
			break_bond = []
			for bidx in br:
				bond = bonds[bidx]
				a1 = bond.GetBeginAtom()
				a2 = bond.GetEndAtom()
				a1n = [n.GetSymbol() for n in a1.GetNeighbors() if n.GetSymbol()=='C']
				a2n = [n.GetSymbol() for n in a2.GetNeighbors() if n.GetSymbol()=='C']
				if len(a1n) == 3 and len(a2n) == 3:
					bridgehead_bond.append(bidx)
				elif len(a1n) == 2 and len(a2n) == 2:
					bridge_bond.append(bidx)
				elif len(a1n) == 3 and len(a2n) == 2:
					bridge_bond.append(bidx)
					break_bond.append(bidx)
				elif len(a1n) == 2 and len(a2n) == 3:
					bridge_bond.append(bidx)
					break_bond.append(bidx)
			#Case1
			if len(bridge_bond)==5 and len(bridgehead_bond)==1:
				#print('Case1', bridge_bond, bridgehead_bond)
				break_atoms = [[bonds[b].GetBeginAtomIdx(),bonds[b].GetEndAtomIdx()] for b in break_bond]
				bridge_atoms = [[bonds[b].GetBeginAtomIdx(),bonds[b].GetEndAtomIdx()] for b in bridge_bond]
				mol_ch = deepcopy(mol)
				atoms_ch = mol_ch.GetAtoms()
				bonds_ch = mol_ch.GetBonds()
				frag = AllChem.MolFromSmiles('CCC')
				frag_atoms = frag.GetAtoms()
				frag_idx =  [fa.GetIdx()+len(atoms_ch) for fa in frag_atoms]
				addbond_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 3] for bs in break_atoms]
				addbond_atoms = [t[0] for t in addbond_atoms]
				remove_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 2] for bs in bridge_atoms]
				remove_atoms = list(sorted(set([t[0] for t in remove_atoms]),reverse=True))
				combine = AllChem.CombineMols(mol_ch,frag)
				atoms_combine = combine.GetAtoms()

				for fi in frag_idx:
					atoms_combine[fi].SetNoImplicit(True)
					atoms_combine[fi].SetNumExplicitHs(2)
				edit_mol = AllChem.EditableMol(combine)
				for ba_ in break_atoms:
					edit_mol.RemoveBond(ba_[0],ba_[1])
				for ba_ in break_atoms:
					if addbond_atoms[0] in ba_:
						edit_mol.AddBond(addbond_atoms[0],frag_idx[0],order=Chem.rdchem.BondType.SINGLE)
					elif addbond_atoms[1] in ba_:
						edit_mol.AddBond(addbond_atoms[1],frag_idx[-1],order=Chem.rdchem.BondType.SINGLE)
				[edit_mol.RemoveAtom(rm) for rm in remove_atoms]
				combine2 = edit_mol.GetMol()
				atoms_cb = combine2.GetAtoms()
				new_frag_idx = [atom.GetIdx() for atom in atoms_cb][-2:]
				nns = [[n.GetIdx() for n in atoms_cb[nfi].GetNeighbors() if n.GetIdx() not in new_frag_idx] for nfi in new_frag_idx]
				nns = [t[0] for t in nns]
				for n1,n2 in zip(new_frag_idx,nns):
					check = list(set([tmp.GetBondType() for tmp in atoms_cb[n2].GetBonds()]))
					if check == [Chem.rdchem.BondType.SINGLE]:
						bond = combine2.GetBondBetweenAtoms(n1,n2)
						bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
						atoms_cb[n1].SetNoImplicit(True)
						atoms_cb[n1].SetNumExplicitHs(1)
				
				mols.append(combine2)
				#combine2 = molwith_idx(combine2)
				#combine2 = AllChem.AddHs(combine2)
				#AllChem.Compute2DCoords(combine2)
				#Chem.Draw.MolToFile(combine2,'check.png', size=(800,800))

				
			#Case2
			elif len(bridge_bond)==4 and len(bridgehead_bond)==2:
				#print('Case2', bridge_bond, bridgehead_bond)
				break_atoms = [[bonds[b].GetBeginAtomIdx(),bonds[b].GetEndAtomIdx()] for b in break_bond]
				bridge_atoms = [[bonds[b].GetBeginAtomIdx(),bonds[b].GetEndAtomIdx()] for b in bridge_bond]
				mol_ch = deepcopy(mol)
				atoms_ch = mol_ch.GetAtoms()
				bonds_ch = mol_ch.GetBonds()
				frag = AllChem.MolFromSmiles('CC')
				frag_atoms = frag.GetAtoms()
				frag_idx =  [fa.GetIdx()+len(atoms_ch) for fa in frag_atoms]
				addbond_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 3] for bs in break_atoms]
				addbond_atoms = [t[0] for t in addbond_atoms]
				remove_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 2] for bs in bridge_atoms]
				remove_atoms = list(sorted(set([t[0] for t in remove_atoms]),reverse=True))
				combine = AllChem.CombineMols(mol_ch,frag)
				atoms_combine = combine.GetAtoms()

				for fi in frag_idx:
					atoms_combine[fi].SetNoImplicit(True)
					atoms_combine[fi].SetNumExplicitHs(2)
				edit_mol = AllChem.EditableMol(combine)
				for ba_ in break_atoms:
					edit_mol.RemoveBond(ba_[0],ba_[1])
				for ba_ in break_atoms:
					if addbond_atoms[0] in ba_:
						edit_mol.AddBond(addbond_atoms[0],frag_idx[0],order=Chem.rdchem.BondType.SINGLE)
					elif addbond_atoms[1] in ba_:
						edit_mol.AddBond(addbond_atoms[1],frag_idx[-1],order=Chem.rdchem.BondType.SINGLE)
				[edit_mol.RemoveAtom(rm) for rm in remove_atoms]
				combine2 = edit_mol.GetMol()
				atoms_cb = combine2.GetAtoms()
				new_frag_idx = [atom.GetIdx() for atom in atoms_cb][-2:]
				nns = [[n.GetIdx() for n in atoms_cb[nfi].GetNeighbors() if n.GetIdx() not in new_frag_idx] for nfi in new_frag_idx]
				nns = [t[0] for t in nns]
				for n1,n2 in zip(new_frag_idx,nns):
					check = list(set([tmp.GetBondType() for tmp in atoms_cb[n2].GetBonds()]))
					if check == [Chem.rdchem.BondType.SINGLE]:
						bond = combine2.GetBondBetweenAtoms(n1,n2)
						bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
						atoms_cb[n1].SetNoImplicit(True)
						atoms_cb[n1].SetNumExplicitHs(1)

				mols.append(combine2)

			#Case3
			elif len(bridge_bond)==3 and len(bridgehead_bond)==3:
				break_atoms = [[bonds[b].GetBeginAtomIdx(),bonds[b].GetEndAtomIdx()] for b in break_bond]
				mol_ch = deepcopy(mol)
				atoms_ch = mol_ch.GetAtoms()
				bonds_ch = mol_ch.GetBonds()
				frag = AllChem.MolFromSmiles('C')
				frag_atoms = frag.GetAtoms()
				frag_idx =  [fa.GetIdx()+len(atoms_ch) for fa in frag_atoms]
				addbond_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 3] for bs in break_atoms]
				addbond_atoms = [t[0] for t in addbond_atoms]
				remove_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 2] for bs in break_atoms]
				remove_atoms = [t[0] for t in remove_atoms]
				combine = AllChem.CombineMols(mol_ch,frag)
				atoms_combine = combine.GetAtoms()
				for fi in frag_idx:
					atoms_combine[fi].SetNoImplicit(True)
					atoms_combine[fi].SetNumExplicitHs(2)
					
				edit_mol = AllChem.EditableMol(combine)
				for ba_ in break_atoms:
					edit_mol.RemoveBond(ba_[0],ba_[1])
				for ba_ in break_atoms:
					if addbond_atoms[0] in ba_:
						edit_mol.AddBond(addbond_atoms[0],frag_idx[0],order=Chem.rdchem.BondType.SINGLE)
					elif addbond_atoms[1] in ba_:
						edit_mol.AddBond(addbond_atoms[1],frag_idx[-1],order=Chem.rdchem.BondType.SINGLE)
				[edit_mol.RemoveAtom(rm) for rm in remove_atoms]
				combine2 = edit_mol.GetMol()
				atoms_cb = combine2.GetAtoms()
				mols.append(combine2)
			
			#Case4
			elif len(bridge_bond)==2 and len(bridgehead_bond)==4:
				break_atoms = [[bonds[b].GetBeginAtomIdx(),bonds[b].GetEndAtomIdx()] for b in break_bond]
				mol_ch = deepcopy(mol)
				atoms_ch = mol_ch.GetAtoms()
				bonds_ch = mol_ch.GetBonds()
				addbond_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 3] for bs in break_atoms]
				addbond_atoms = [t[0] for t in addbond_atoms]
				remove_atoms = [[s for s in bs if len(atoms_ch[s].GetNeighbors()) == 2] for bs in break_atoms]
				remove_atoms = list(set([t[0] for t in remove_atoms]))

				[atoms_ch[ri].SetIsAromatic(False) for ri in remove_atoms]
				[[b.SetBondType(Chem.rdchem.BondType.SINGLE) for b in atoms_ch[ri].GetBonds()] for ri in remove_atoms]
				[[b.SetIsAromatic(False) for b in atoms_ch[ri].GetBonds()] for ri in remove_atoms]
				btype = [[bb.GetBondType() for bb in atoms_ch[ai].GetBonds()] for ai in addbond_atoms]
				for bi, bt in enumerate(btype):
					if list(set(bt)) == [Chem.rdchem.BondType.SINGLE]:
						tidx = addbond_atoms[bi]
						atoms_ch[tidx].SetNoImplicit(True)
						atoms_ch[tidx].SetNumExplicitHs(1)

				edit_mol = AllChem.EditableMol(mol_ch)
				for ba_ in break_atoms:
					edit_mol.RemoveBond(ba_[0],ba_[1])
				edit_mol.AddBond(addbond_atoms[0],addbond_atoms[1],order=Chem.rdchem.BondType.SINGLE)
				
				[edit_mol.RemoveAtom(rm) for rm in remove_atoms]
				combine2 = edit_mol.GetMol()
				atoms_cb = combine2.GetAtoms()
				mols.append(combine2)


	return mols
