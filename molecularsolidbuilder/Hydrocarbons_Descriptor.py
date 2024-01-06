import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase.io import read, write
import numpy as np
import os, sys, glob, subprocess
from itertools import combinations
from collections import Counter

def HydroCarbon_prop(m):
	formula = AllChem.CalcMolFormula(m)
	#print(formula)

	atoms = m.GetAtoms()
	hybrid = [atom.GetHybridization() for atom in atoms]
	num_sp2 = hybrid.count(Chem.rdchem.HybridizationType.SP2)
	num_sp3 = hybrid.count(Chem.rdchem.HybridizationType.SP3)
	hybridization = {"sp2":num_sp2, "sp3":num_sp3}
	#print(hybridization)

	m_h = AllChem.AddHs(m)
	atoms_h = m_h.GetAtoms()
	nns = [atom.GetSymbol() + "["+ "".join([n.GetSymbol() for n in atom.GetNeighbors() ])+"]" for atom in atoms_h]
	connectivity = {"C[CCC]":0,
					"C[HHH]":0,
					"C[CCH]":0,"C[CCHH]":0,"H[C]":0
	               }
	for key, value in connectivity.items():
		num = nns.count(key)
		connectivity[key] = num
	#print(connectivity)

	ring  = m.GetRingInfo()
	ring_members = ring.AtomRings()
	num_ring = [len(r) for r in ring_members]
	ring_nums = dict(Counter(num_ring))
	#print(ring_nums)
	
	prop = {'formula':formula,'hybridization':hybridization,'connectivity':connectivity,'ring_nums':ring_nums}
	return prop
