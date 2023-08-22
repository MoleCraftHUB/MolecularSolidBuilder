import os, sys, subprocess, glob
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from copy import deepcopy
from .PDBfile_importer import PDBImageFileToMols

def molwith_idx(mol):
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(atom.GetIdx())
	return mol


def BreakCrosslink(pdb_filename):
    f = open(pdb_filename,'r')
    lines = f.readlines()
    end_index = [l for l in range(len(lines)) if 'END' in lines[l]]
    start = 0
    ms = []
    for i in range(len(end_index)):
        end = end_index[i] + 1
        pdb_block = lines[start:end]
        start = end
        pdb_block_str = "".join(pdb_block)
        m = AllChem.MolFromPDBBlock(pdb_block_str,removeHs=False)
        #m = AllChem.RemoveHs(m, updateExplicitCount=True)
        ms.append(m)

    for ai, m in enumerate(ms):
        #m = AllChem.MolFromPDBFile('mol_1.pdb',removeHs=False)
        atoms0 = m.GetAtoms()

        ri = m.GetRingInfo()
        ring_members = ri.AtomRings()

        rmem = []
        for rm in ring_members:
            for t in rm:
                rmem.append(t)

        crosslink_atoms = []
        atoms = m.GetAtoms()
        for atom in atoms:
            flag1 = atom.GetIdx() not in rmem
            flag2 = (atom.GetSymbol() == 'O') and len([n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']) == 2
            flag3 = (atom.GetSymbol() == 'S') and len([n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']) == 2
            flag4 = (atom.GetSymbol() == 'S') and len([n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']) == 2
            flag5 =  len([n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']) != 1
            if flag1 and (flag2 or flag3 or flag4) and flag5:
                crosslink_atoms.append(atom.GetIdx())

        highlight_atoms = [atom.GetIdx() for atom in atoms if atom.GetIdx() not in crosslink_atoms]

        highlight_bonds = []
        bonds = m.GetBonds()
        for bond in bonds:
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in highlight_atoms and a2 in highlight_atoms:
                highlight_bonds.append(bond.GetIdx())
        crosslink_bonds = [bond.GetIdx() for bond in bonds if bond.GetIdx() not in highlight_bonds]
        crosslink_bonds_b = [bonds[bi] for bi in crosslink_bonds]

        valence_atom_idx = []
        for i, bond2break in enumerate(crosslink_bonds_b):
            a1 = bond2break.GetBeginAtomIdx()
            a2 = bond2break.GetEndAtomIdx()
            valence_atom_idx += [a1,a2]

        #AllChem.Compute2DCoords(m)
        #Chem.rdCoordGen.AddCoords(m)
        #Draw.MolToFile(m,'./test.png',size=(800,800),highlightAtoms=highlight_atoms,highlightBonds=highlight_bonds)

        #print(len(crosslink_bonds))
        if len(crosslink_bonds) > 0:
            new_m = Chem.FragmentOnBonds(m,crosslink_bonds, addDummies=True)
            new_ms = Chem.GetMolFrags(new_m, asMols=True)
            #new_m = molwith_idx(new_m)
            #AllChem.Compute2DCoords(new_m)
            #Draw.MolToFile(new_m,'new_test.png',size=(2000,2000))
            #subprocess.call('imgcat new_test.png',shell=True)
            pdb_string1 = ""
            npdb1 = 0

            for i, nm in enumerate(new_ms):     
                nm_atoms = nm.GetAtoms()
                for atom in nm_atoms:
                    if atom.GetSymbol() == '*':
                        atom.SetAtomicNum(1)

                #print([atom.GetSymbol() for atom in nm_atoms])
                #AllChem.Compute2DCoords(nm)
                #Draw.MolToFile(nm,'new_test.png',size=(500,500))
                #subprocess.call('imgcat new_test.png',shell=True)
                AllChem.EmbedMolecule(nm)
                AllChem.MMFFOptimizeMolecule(nm)
                syms = [atom.GetSymbol() for atom in nm.GetAtoms()]
                msyms = [s for s in syms if s != 'H']
                if len(msyms) > 2:
                    pdb = AllChem.MolToPDBBlock(nm)
                    pdb_string1 += pdb
                    npdb1 +=1

            print(npdb1,'fragment')
            with open('frag1_%d.pdb' % ai,'w') as f1:
                f1.write(pdb_string1)

        else:

            pdb_string2 = ""
            npdb2 = 0
            print('no ring')
            AllChem.EmbedMolecule(m)
            AllChem.MMFFOptimizeMolecule(m)
            syms = [atom.GetSymbol() for atom in m.GetAtoms()]
            msyms = [s for s in syms if s != 'H']
            if len(msyms) > 2:
                pdb = AllChem.MolToPDBBlock(nm)
                pdb_string2 += pdb
                npdb2 += 1

            print(npdb2,'no fragment')
            with open('frag2_%d.pdb' % ai,'w') as f2:
                f2.write(pdb_string2)
    return 