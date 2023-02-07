import numpy as np
import os, sys, glob, subprocess
from ase import Atoms, Atom
from ase.io import read, write
from ase.data import atomic_masses,atomic_numbers
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from copy import deepcopy

def read_dump_custom(dumpfilename):
    
    f = open(dumpfilename,'r')
    lines = f.readlines()
    start = [i for i, line in enumerate(lines) if 'ITEM: TIMESTEP' in line ]
    for j in range(len(start)-1):
        s = start[j]
        e = start[j+1]
        structure = lines[s:e]
        timestep = structure[1]
        Natoms = structure[3]
        cell = structure[4:8]
        info = structure[8].split()
        symbol_position = info.index('element')-2
        x_ = info.index('x')-2
        y_ = info.index('y')-2
        z_ = info.index('z')-2
        atom_lines = np.array([sl.split() for sl in structure[9:]] ) 
        syms = atom_lines[:,symbol_position]
        positions = atom_lines[:,x_:z_+1].astype(float)
        structure_str = "".join(structure)

    #Return only the last image
    return syms, positions, structure_str



def ReaxFFminimize(pdb_file,path,ncores=12,lmp_path='/home/8py/apps/lammps-stable_3Mar2020/src/lmp_mpi'):
    base_dir = os.getcwd()
    energy = 0
    images = read(pdb_file,index=":")
    atoms = images[0]
    atoms.set_cell([100,100,100])
    atoms.center()
    atoms.write(path+'/molecules.data',format='lammps-data',units='real',atom_style='charge')
    elements = sorted(list(set(atoms.get_chemical_symbols())))
    element_string = " ".join(elements)

    symbols = atoms.get_chemical_symbols()
    masses = [atomic_masses[atomic_numbers[s]] for s in symbols]
    tot_mass = sum(masses)
    Na = 6.023*10**(23)
    d = 1.1
    vol = ( tot_mass / d ) * (10**24) / Na
    length = vol**(1/3)
    #print(length)

    fm = open(path+'/mass.data','w')
    for i, e in enumerate(elements):
        an = atomic_numbers[e]
        fm.write("mass		%d %5.4f\n" % (i+1,atomic_masses[an]))
    fm.close()

    fp = open(path+'/pair.data','w')
    fp.write('pair_style      reax/c NULL safezone 8.0 mincap 1000\n')
    fp.write('pair_coeff      * * ffield.reax %s\n\n' % element_string )
    fp.close()

    fd = open(path+'/dump_elem.data','w')
    fd.write('dump_modify     dp element %s sort id\n\n' % element_string)
    fd.close()
    os.chdir(path)
    subprocess.call('module purge',shell=True)
    subprocess.call('module load PE-intel/3.0',shell=True)
    subprocess.call('mpiexec -np %d %s < in_min.reaxc > out.reax' % (ncores,lmp_path),shell=True)
    p = subprocess.Popen(["grep","lenergy","out.reax"],stdout=subprocess.PIPE)
    out, err = p.communicate()
    energy = float( str(out).split()[-1].replace("\\n'","") )
    syms, positions, structure_str = read_dump_custom('min.dump')


    os.chdir(base_dir)
    return energy, syms, positions, structure_str


