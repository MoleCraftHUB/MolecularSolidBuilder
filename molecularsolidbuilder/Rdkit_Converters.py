from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase.io import read, write
from ase import Atoms, Atom
import numpy as np
import os, sys, glob, subprocess
from ase.visualize import view

def molwith_idx(mol):
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(atom.GetIdx())
	return mol

def isRingAromatic(mol, bondRing):
	for id in bondRing:
		if not mol.GetBondWithIdx(id).GetIsAromatic():
			return False
		return True

def PDBimport_PreserveHs(pdb_file):

	m = AllChem.MolFromPDBFile(pdb_file,removeHs=False)
	m2 = AllChem.RemoveHs(m, updateExplicitCount=True)
	atoms2 = m2.GetAtoms()

	[atom.SetNumRadicalElectrons(1) for atom in atoms2 if atom.GetSymbol() == 'C' and len(atom.GetNeighbors())+atom.GetNumExplicitHs()==3]
	[atom.SetNumRadicalElectrons(1) for atom in atoms2 if atom.GetSymbol() == 'N' and len(atom.GetNeighbors())+atom.GetNumExplicitHs()==2]
	[atom.SetNumRadicalElectrons(1) for atom in atoms2 if atom.GetSymbol() == 'O' and len(atom.GetNeighbors())+atom.GetNumExplicitHs()==1]

	[atom.SetNumRadicalElectrons(2) for atom in atoms2 if atom.GetSymbol() == 'C' and len(atom.GetNeighbors())+atom.GetNumExplicitHs()==2]
	[atom.SetNumRadicalElectrons(2) for atom in atoms2 if atom.GetSymbol() == 'N' and len(atom.GetNeighbors())+atom.GetNumExplicitHs()==1]

	ri = m2.GetRingInfo()          #Get Ring information
	ring_members = ri.AtomRings()  #Get members of Rings
	ring_members = [r for r in ring_members if len(r) == 6]
	for r in ring_members:
		arom = [True if atoms2[i].GetNumRadicalElectrons() == 1 else False for i in r]
		flag = all(arom)
		if flag:
			for rii in r:
				atom = atoms2[rii]
				atom.SetIsAromatic(True)
				atom.UpdatePropertyCache()
	bonds_r = ri.BondRings()
	bonds = m2.GetBonds()
	for bond_idx in bonds_r:
		for bi in bond_idx:
			bond = bonds[bi]
			a = bond.GetBeginAtom()
			b = bond.GetEndAtom()
			ah = a.GetIsAromatic()
			bh = b.GetIsAromatic()
			if ah and bh:
				bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
				a.SetNumRadicalElectrons(0)
				b.SetNumRadicalElectrons(0)
				a.UpdatePropertyCache()
				b.UpdatePropertyCache()
				bonds = m2.GetBonds()
			else:
				bond.SetBondType(Chem.rdchem.BondType.SINGLE)


	bonds = m2.GetBonds()
	for bond in bonds:
		a = bond.GetBeginAtom()
		b = bond.GetEndAtom()
		ar = a.GetNumRadicalElectrons()
		br = b.GetNumRadicalElectrons()
		if ar == 1 and br == 1:
			bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
			a.SetNumRadicalElectrons(0)
			b.SetNumRadicalElectrons(0)
			a.UpdatePropertyCache()
			b.UpdatePropertyCache()
			bonds = m2.GetBonds()

	return m2

	#m2 = AllChem.AddHs(m2,explicitOnly=True)

	#AllChem.Compute2DCoords(m2)
	#Chem.Draw.MolToFile(m2,'./test.png',size=(1500,1500),kekulize=True)
	#subprocess.call('imgcat ./test.png', shell=True)

