import sys,os,glob,subprocess,math,random
from ase.io import read, write
from ase.visualize import view
from ase import Atoms, Atom
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdCoordGen, Descriptors, rdMolHash
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdchem import HybridizationType


def lmpdumpfile_cell_info(lmpdumpfile):

    fdump = open(lmpdumpfile,'r')
    lines = fdump.readlines()
    start = 0
    end1 = -1
    cell_start = 0

    # Collect the cell information    
    for i, l in enumerate(lines):
        if 'ITEM: ATOMS' in l:
            start = i+1
        if 'ITEM: BOX' in l:
            cell_start = i+1
            cell = np.array([ll.split() for ll in lines[cell_start:cell_start+3]]).astype(float)

    if cell.shape==(3,3):
        xy = cell[0][2]
        xz = cell[1][2]
        yz = cell[2][2]
        xlo = cell[0][0] - min([0.0,cell[0][2],cell[1][2],cell[0][2]+cell[1][2]])
        xhi = cell[0][1] - max([0.0,cell[0][2],cell[1][2],cell[0][2]+cell[1][2]])
        ylo = cell[1][0] - min([0.0,cell[2][2]])
        yhi = cell[1][1] - max([0.0,cell[2][2]])
        zlo = cell[2][0]
        zhi = cell[2][1]
    else:
        xy = 0
        xz = 0
        yz = 0
        xlo = cell[0][0]
        xhi = cell[0][1]
        ylo = cell[1][0]
        yhi = cell[1][1]
        zlo = cell[2][0]
        zhi = cell[2][1]

    cell_f = [[xlo,xhi], [ylo,yhi], [zlo,zhi]]
    cell_mat = np.array([[xhi-xlo, 0, 0],[xy,yhi-ylo,0],[xz,yz,zhi-zlo]])
    return cell_f, cell_mat

def pdbmols_to_aseatoms(lmpdumpfile,pdb_collect_dir):

    """

    """

    cell_mat, cell_f = lmpdumpfile_cell_info(lmpdumpfile)
    pdb_files = list(sorted(glob.glob('%s/*.pdb' % pdb_collect_dir),key=lambda x:int(x[:-4].split('_')[-1])))
    symbols = []
    pos = []
    combined = []

    for pdb_file in pdb_files:
        atoms = read(pdb_file)
        ss = atoms.get_chemical_symbols()
        pp = atoms.get_positions()
        for s,p in zip(ss,pp):
            combined.append([s,p-np.array([cell_f[0][0],cell_f[1][0],cell_f[2][0]])])
    combined_sorted = sorted(combined,key=lambda x:str(x[0]))

    for c in combined_sorted:
        symbols.append(c[0])
        pos.append(c[1])

    atoms = Atoms(symbols,positions=pos,cell=cell_mat,pbc=[True,True,True])
    #atoms.write('str1.xyz')

    return atoms


def collect_mols_from_lmpdump(lmpdumpfile,pdb_start_dir,systemltfile,pdb_collect_dir='./collected_pdbs'):

    """
    lmpdumpfile format

    """
    cell_mat, cell_f = lmpdumpfile_cell_info(lmpdumpfile)
    # Atom coordinate information

    fdump = open(lmpdumpfile,'r')
    lines = fdump.readlines()
    start = 0
    end1 = -1

    line_pos = lines[start:]
    line_pos_split = np.array([l.split() for l in line_pos if len(l) > 0])
    mol_id = sorted(list(set(line_pos_split[:,1])),key=lambda x:int(x))

    mol_seperated = []
    for i, mid in enumerate(mol_id):
        each_mol = sorted([l for l in line_pos_split if int(l[1]) == int(mid)],key=lambda x:x[2])
        each_mol = np.array(each_mol)
        each_mol[:,4] = each_mol[:,4].astype(float)
        each_mol[:,5] = each_mol[:,5].astype(float)
        each_mol[:,6] = each_mol[:,6].astype(float)
        mol_seperated.append(each_mol.astype(str))

    # Collect molecule id information

    fsystem = open(systemltfile,'r')
    mol_id_ref = [int(line.split('.move')[0].split('new mol_')[-1]) for line in fsystem.readlines() if '= new mol' in line]

    start_pdb = list(sorted(glob.glob('%s/*.pdb' % (pdb_start_dir)),key=lambda x:int(x[:-4].split('_')[-1])))

    new_mol_filenames = []
    for i, mol_coords in enumerate(mol_seperated):
        ref_pdb = open(start_pdb[mol_id_ref[i]-1],'r')
        ref_pdb_lines = ref_pdb.readlines()
        mol_coords_sorted = np.array(sorted(mol_coords,key=lambda x:int(x[0])))
        ref_atom_lines = [ref_atom for ref_atom in ref_pdb_lines if 'HETATM' in ref_atom]
        ref_bond_lines = [ref_atom for ref_atom in ref_pdb_lines if 'HETATM' not in ref_atom]
        cell_mat_flatten = cell_mat.flatten()
        ref_bond_lines[-1] = 'END #3DBOX %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f\n' % cell_mat_flatten

        elements = mol_coords_sorted[:,3]
        coords = mol_coords_sorted[:,4:].astype(float)
        for idx, atom_line in enumerate(coords):
            check = ref_atom_lines[idx]
            string = check[:26]+'%12.3f %7.3f %7.3f ' % (atom_line[0],atom_line[1],atom_line[2])+check[-26:]
            ref_atom_lines[idx] = string
        modified_pdb = ref_atom_lines + ref_bond_lines

        new_pdb_string = "".join(modified_pdb)
        f = open('%s/mol_%d.pdb' % (pdb_collect_dir,mol_id_ref[i]),'w')
        f.write(new_pdb_string)
        new_mol_filenames.append('%s/mol_%d.pdb' % (pdb_collect_dir,mol_id_ref[i]))
        print('%s/mol_%d.pdb' % (pdb_collect_dir,mol_id_ref[i]))
        f.close()

    return new_mol_filenames

def convert_pdb_to_gaff_mol2(pdb_start_dir='1_split_pdb',mol2_collect_dir='2_split_mol2_typing'):

    mol_files_path = sorted(glob.glob('%s/*.pdb' % pdb_start_dir),key=lambda x:int(x[:-4].split('_')[-1]))
    if not os.path.exists('%s' % mol2_collect_dir):
        os.mkdir('%s' % mol2_collect_dir)
    else:
        subprocess.call('rm %s/*.mol2' % mol2_collect_dir,shell=True)

    mols = [AllChem.MolFromPDBFile(mol,removeHs=False) for mol in mol_files_path]

    t = []
    for ii, mol_H in enumerate(mols):
        mol_filename = mol_files_path[ii]
        cmd = 'obabel -ipdb %s -omol2 -Otmp.mol2 --partialcharge gasteiger' % mol_filename
        subprocess.call(cmd,shell=True)
        print('==============================')
        print(ii,mol_filename)
        with open('./tmp.mol2','r') as f0:
            lines = f0.readlines()
            charges = [line.split()[-1] for line in lines if 'UN' in line]
            check = [line.split() for line in lines]

        split = 0
        for i, c in enumerate(check):
            if (len(c) > 0) and ("@<TRIPOS>BOND" in c[0]):
                split = i
        header = check[:6]
        atom_sect = check[6:split]
        bond_sect = check[split:]

        atoms = mol_H.GetAtoms()
        ri = mol_H.GetRingInfo()
        ring_data = ri.AtomRings()
        idx  = [atom.GetIdx() for atom in atoms]
        chem = [atom.GetSymbol() for atom in atoms]
        arom = [atom.GetIsAromatic() for atom in atoms]
        hybrid=[atom.GetHybridization() for atom in atoms]
        gaff_types = [atom.GetSymbol() for atom in atoms]
        for a, atom in enumerate(atoms):
            sidx = atom.GetIdx()
            rsize_check = list(sorted([len(rd) for rd in ring_data if sidx in rd]))
            sym = atom.GetSymbol()
            hyb = atom.GetHybridization()
            arm = atom.GetIsAromatic()
            nn = atom.GetNeighbors()
            nn_sym = [n.GetSymbol() for n in nn]
            nn_sym_H = [n.GetSymbol() for n in nn if n.GetSymbol() == 'H']
            nn_count = len(nn_sym)
            nn_H_count = len(nn_sym_H)
            nn_arom = [n.GetIsAromatic() for n in nn]
            nn_hybrid = [n.GetHybridization() for n in nn]
            nn_idx = [n.GetIdx() for n in nn]
            nn_bt = [mol_H.GetBondBetweenAtoms(sidx,n.GetIdx()).GetBondType() for n in nn]

            #print(sym,hyb,arm,nn_sym,nn_sym_H,nn_arom,nn_hybrid,nn_bt,rsize_check)
            #Hydrogens
            if (sym=='H') and (nn_sym==['C']) and (nn_arom==[True]):
                gaff_types[a] = 'ha'
            elif (sym=='H') and (nn_sym==['C']) and (nn_arom==[False]) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP3]):
                gaff_types[a] = 'hc'
            elif (sym=='H') and (nn_sym==['C']) and (nn_arom==[False]) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP2]):
                gaff_types[a] = 'hc'
            elif (sym=='H') and (nn_sym==['O']):
                gaff_types[a] = 'ho'
            elif (sym=='H') and (nn_sym==['N']):
                gaff_types[a] = 'hn'
            elif (sym=='H') and (nn_sym==['S']):
                gaff_types[a] = 'hs'
            #Carbons
            elif (sym=='C') and (arm==True) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (6 in rsize_check) and (nn_H_count==0):
                gaff_types[a] = 'ca'
            elif (sym=='C') and (arm==True) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (6 in rsize_check) and (nn_H_count==1):
                gaff_types[a] = 'ca'
            elif (sym=='C') and (arm==True) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (5 in rsize_check) and (nn_H_count==1):
                gaff_types[a] = 'ca'
            elif (sym=='C') and (arm==True) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (5 in rsize_check) and (nn_H_count==0):
                gaff_types[a] = 'ca'
            elif (sym=='C') and (arm==True) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (5 in rsize_check and 6 in rsize_check):
                gaff_types[a] = 'cb'
            elif (sym=='C') and (arm==False) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2):
                gaff_types[a] = 'c2'
            elif (sym=='C') and (arm==False) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP3):
                gaff_types[a] = 'c3'
            #Oxygens
            elif (sym=='O') and (arm==False) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP2]) and (nn_bt==[rdkit.Chem.rdchem.BondType.DOUBLE]):
                gaff_types[a] = 'o'
            elif (sym=='O') and (arm==True) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP2,rdkit.Chem.rdchem.HybridizationType.SP2]) and (nn_bt==[rdkit.Chem.rdchem.BondType.AROMATIC,rdkit.Chem.rdchem.BondType.AROMATIC]):
                gaff_types[a] = 'o'
            elif (sym=='O') and (arm==False) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and ((nn_sym==['C','H']) or (nn_sym==['H','C'])):
                gaff_types[a] = 'oh'
            elif (sym=='O') and (arm==False) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP3) and ((nn_sym==['C','H']) or (nn_sym==['H','C'])):
                gaff_types[a] = 'oh'
            elif (sym=='O') and (arm==False) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP3) and ((nn_sym==['N','H']) or (nn_sym==['H','N'])):
                gaff_types[a] = 'oh'
            elif (sym=='O') and (arm==False) and (nn_sym==['C','C']) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP3,rdkit.Chem.rdchem.HybridizationType.SP3]):
                gaff_types[a] = 'os'
            elif (sym=='O') and (arm==False) and (nn_sym==['C','C']) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP2,rdkit.Chem.rdchem.HybridizationType.SP3]):
                gaff_types[a] = 'os'
            elif (sym=='O') and (arm==False) and (nn_sym==['C','C']) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP3,rdkit.Chem.rdchem.HybridizationType.SP2]):
                gaff_types[a] = 'os'
            elif (sym=='O') and (arm==False) and (nn_sym==['C','C']) and (nn_hybrid==[rdkit.Chem.rdchem.HybridizationType.SP2,rdkit.Chem.rdchem.HybridizationType.SP2]):
                gaff_types[a] = 'os'
            #Nitrogens
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (arm==True):
                gaff_types[a] = 'n'
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (arm==True):
                gaff_types[a] = 'n'
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (arm==False) and (nn_count==3) and (rsize_check==[6]):
                gaff_types[a] = 'nt'
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (arm==False) and (nn_count==2) and (rsize_check==[6]):
                gaff_types[a] = 'nb'
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2) and (arm==False) and (nn_sym==['C','C','C']) and (rsize_check==[5]):
                gaff_types[a] = 'na'
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP3) and (arm==False) and (nn_sym==['C','C','C']):
                gaff_types[a] = 'n3'
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP3) and (arm==False) and ('N' in nn_sym) and ('C' in nn_sym) and (nn_count==3):
                gaff_types[a] = 'n3'
            elif (sym=='N') and (hyb==rdkit.Chem.rdchem.HybridizationType.SP3) and (arm==False) and ('O' in nn_sym) and ('C' in nn_sym) and (nn_count==3):
                gaff_types[a] = 'n3'

            #Sulfurs
            elif (sym=='S') and (arm==False) and (nn_sym==['C','H']):
                gaff_types[a] = 'sh'
            elif (sym=='S') and (arm==False) and (nn_sym==['H','C']):
                gaff_types[a] = 'sh'
            elif (sym=='S') and (arm==False) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP3) and (nn_sym==['C','C']):
                gaff_types[a] = 'ss'
            elif (sym=='S') and (arm==True) and (nn_sym==['C','C']) and (hyb==rdkit.Chem.rdchem.HybridizationType.SP2):
                gaff_types[a] = 's2'
            #Error
            else:
                print(sym,hyb,arm,nn_sym,nn_sym_H,nn_arom,nn_hybrid,nn_bt,rsize_check)
                sys.exit()

        print(gaff_types)
        print('==============================\n')

        m = mol_H.GetConformers()
        pos = m[0].GetPositions()
        check_sym = []
        atom_names = []
        for i, c in enumerate(chem):
            a = len([t for t in check_sym if t ==c])
            check_sym.append(c)
            atom_name = c+str(a+1)
            atom_names.append(atom_name)

        Chem.Kekulize(mol_H)
        bonds = mol_H.GetBonds()
        bond_chem = [[chem[b.GetBeginAtomIdx()],chem[b.GetEndAtomIdx()]] for b in bonds]
        bond_ref = []
        bond_type = []
        for i, b in enumerate(bonds):
            aidx1 = b.GetBeginAtomIdx()
            aidx2 = b.GetEndAtomIdx()
            achem1 = atom_names[aidx1]
            achem2 = atom_names[aidx2]
            bond_ref.append([achem1,achem2])
            if b.GetBondTypeAsDouble() == 1.5:
                bond_type.append('ar')
            elif b.GetBondTypeAsDouble() == 1:
                bond_type.append('1')
            elif b.GetBondTypeAsDouble() == 2.0:
                bond_type.append('2')

        for i, atom in enumerate(atoms):
            atom_name = atom_names[i]
            gaff_type = gaff_types[i]
            charge = charges[i]
            px = pos[i][0]
            py = pos[i][1]
            pz = pos[i][2]

        bonds = mol_H.GetBonds()
        bond_idx = [[b.GetBeginAtomIdx(),b.GetEndAtomIdx()] for b in bonds]
        bond_names = [[atom_names[b.GetBeginAtomIdx()],atom_names[b.GetEndAtomIdx()]] for b in bonds]


        f2 = open('%s/mol_%d.mol2' % (mol2_collect_dir,ii+1),'w')
        for i, mol2_header_line in enumerate(header):
            f2.write('%s\n'% " ".join(mol2_header_line))
        f2.write('%s\n' % "".join(atom_sect[0]))

        for i, mol2_atom_line in enumerate(atom_sect[1:]):
            mol2_atom_line[5] = gaff_types[i]
            f2.write("%7s%5s%14s%10s%10s%4s%8s%5s%10s\n" % (mol2_atom_line[0],mol2_atom_line[1],mol2_atom_line[2],mol2_atom_line[3],mol2_atom_line[4],mol2_atom_line[5],mol2_atom_line[6],mol2_atom_line[7],mol2_atom_line[8]))

        f2.write('%s\n' % "".join(bond_sect[0]))
        for i, mol2_bond_line in enumerate(bond_sect[1:]):
            mol2_bond_line[-1] = bond_type[i]
            f2.write("%6s%6s%6s%6s\n" %(mol2_bond_line[0],mol2_bond_line[1],mol2_bond_line[2],mol2_bond_line[3]))

        f2.close()
        subprocess.call('rm tmp.mol2',shell=True)

    return

def am1bcc_charge_antechamber(mol2_files_dir,restart=False,cal_dir='3_mol2s',slurm=True,job_queue={'-A':'ccsd',
                                                                                                   '-p':'batch',
                                                                                                   '--nodes':1,
                                                                                                   '--ntasks':1,
                                                                                                   '-t':'48:00:00',
                                                                                                   }):

    if not os.path.exists('%s' % cal_dir):
        os.mkdir('%s' % cal_dir)
    else:
        if restart:
            #examine calculations are done
            print('check')
        else:
            subprocess.call('rm -r %s' % cal_dir,shell=True)

    base_dir = os.getcwd()

    mol2_files_path = sorted(glob.glob('%s/*.mol2' % mol2_files_dir),key=lambda x:int(x[:-5].split('_')[-1]))
    for i, mol2_file in enumerate(mol2_files_path):
        subdir = '%s/mol2s_antechamber_%d' % (cal_dir,i+1)
        if not os.path.exits(subdir):
            os.mkdir(subdir)

        if slurm:
            sf = open('%s/submit.slurm' % (subdir,i+1),'w')
            sf.write("#!/bin/bash\n")
            sf.write("#SBATCH -A %s\n" % (job_queue['-A']))
            sf.write("#SBATCH -p %s\n" % (job_queue['-p']))
            sf.write("#SBATCH --mem 0\n")
            sf.write("#SBATCH --nodes=%s\n" % (job_queue['--nodes']))
            sf.write("#SBATCH --ntasks=%s\n" % (job_queue['--ntasks']))
            sf.write("#SBATCH -t=%s\n" % (job_queue['-t']))
            sf.write("#SBATCH -J mol2s_antechamber_%d\n" % (i+1))
            sf.write("cd $SLURM_SUBMIT_DIR\n")
            sf.write("cd antechamber -i ../../%s -fi mol2 -o ./new_mol_%d.mol2 -fo mol2 -c bcc -s 2 -pl 15\n" % (mol2_file,i+1))
            os.chdir(subdir)
            subprocess.call('sbatch submit.slurm',shell=True)
            os.chdir(base_dir)
        else:
            #serial calculation
            print('complete serial run part')

    return        


def convert_mol2_to_lt(mol2_cal_dir='3_mol2s',mol_lt_dir='4_lts'):

    mol2_files = sorted(glob.glob('%s/mol2s_antechamber_*/new_mol*.mol2' % mol2_cal_dir),key=lambda x:int(x[:-5].split('_')[-1]))
    if not os.path.exists(mol_lt_dir):
        os.mkdir(mol_lt_dir)    
    else:
        subprocess.call('rm %s/*.lt' % mol_lt_dir,shell=True)
    
    for i, path in enumerate(mol2_files):
        fname = path.split('/')[-1]
        cmd = 'mol22lt.py --in %s --out 4_lts/%s --name %s --ff GAFF --ff-file "gaff.lt"' % (path,fname[4:-5]+'.lt',fname[4:-5])
        subprocess.call(cmd,shell=True)
        print('convert '+fname)

    return

def generate_systemlt_withcoords(lmpdumpfile,mol_lt_dir='4_lts',lt_dir='moltemplate_files'):

    mol_lt_files = sorted(glob.glob('%s/mol*.lt' % mol_lt_dir),key=lambda x:int(x[:-3].split('_')[-1]))

    cell_mat, cell_f = lmpdumpfile_cell_info(lmpdumpfile)

    strings = ""
    for i, mol_lt_file in enumerate(mol_lt_files):
        strings += "import %s\n" % mol_lt_file.split('/')[-1]

    for i, mol_lt_file in enumerate(mol_lt_files):
        strings += "mol%d = new %s\n" % (i+1,mol_lt_file.split('/')[-1][:-3])
    
    xy = cell_mat[1][0]
    xz = cell_mat[2][0]
    yz = cell_mat[2][1]
    strings += '\n\nwrite_once("Data Boundary") {\n'
    strings += "%4.3f %4.3f  xlo xhi\n" % (0.0,cell_f[0])
    strings += "%4.3f %4.3f  ylo yhi\n" % (0.0,cell_f[1])
    strings += "%4.3f %4.3f  zlo zhi\n" % (0.0,cell_f[2])
    strings += "%4.3f %4.3f %4.3f xy xz yz\n" % (xy,xz,yz)
    strings += "}\n"

    if not os.path.exists(lt_dir):
        os.mkdir(lt_dir)
    else:
        subprocess.call('rm %s/*.lt' % lt_dir,shell=True)

    f = open('%s/system.lt' % lt_dir,'w')
    f.write(strings)
    f.close()

    for i, mol_lt_file in enumerate(mol_lt_files):
        subprocess.call('cp %s %s' % (mol_lt_file,lt_dir),shell=True)




###############

# Currently working.. 
# /nfs/data/ccsd/proj-shared/pilsun/c4ward_project3_large_scale_pitch_model/small_dft/1_shifted_layer_gaff
def generate_systemlt_SL(lmpdumpfile,pdb_start_dir='1_split_pdb', mol_lt_dir='4_lts',lt_dir='moltemplate_files',random_arrangement=True):

    cell_mat, cell_f = lmpdumpfile_cell_info(lmpdumpfile)
    mol_pdb_files = sorted(glob.glob('%s/*.pdb' % pdb_start_dir),key=lambda x:int(x[:-4].split('_')[-1]))
    mol_lt_files = sorted(glob.glob('%s/mol*.lt' % mol_lt_dir),key=lambda x:int(x[:-3].split('_')[-1]))

    box_sizes = []
    for i, pdb_file in enumerate(mol_pdb_files):
        box_size = np.array(open(pdb_file,'r').readlines()[-1].split()[2:])
        box_size = box_size.astype(float)
        box_sizes.append(box_size) #3x3 matrix...
    box_sizes = np.array(box_sizes)
    names = [mol_lt_name[:-3] for mol_lt_name in mol_lt_files]
    biggest_box_size = [max(box_sizes[:,0]),max(box_sizes[:,1]),max(box_sizes[:,2])]

    x_num = 2
    y_num = 2
    z_num = 2
    nmols = 1000

    anglex1 = 42
    angley1 = 30

    anglex2 = 42
    angley2 = 30


    print(abs( math.cos(math.radians(anglex1)) ), abs( math.cos(math.radians(angley1)) ))
    x_dist = biggest_box_size[0] * abs( math.cos(math.radians(angley1)) ) * 1.1
    y_dist = biggest_box_size[1] * abs( math.cos(math.radians(anglex1)) ) * 1.1
    z_dist1 = 5


    a = 0

    strings = ""
    for i, mol_lt_file in enumerate(mol_lt_files):
        strings += "import %s\n" % mol_lt_file.split('/')[-1]

    #names_increased1 = [name.split('/')[-1] for name in names][:-1]*x_num*y_num*z_num #Dimer crosslinked
    names_increased2 = [name.split('/')[-1] for name in names]*x_num*y_num*z_num #Monomer

    names_increased_cut = names_increased2[:nmols]
    if random_arrangement:
        random.shuffle(names_increased_cut)

    x_num2 = x_num
    y_num2 = y_num
    z_num2 = z_num

    x_cell = x_dist*x_num2
    y_cell = y_dist*y_num2
    z_cell = z_dist1*z_num2

    for i_x in range(0,x_num2,1):
        for i_y in range(0,y_num2,2):
            for i_z in range(z_num2):
                x_pos = x_dist*i_x - (x_dist/y_num) * i_y
                y_pos = y_dist*i_y
                z_pos = z_dist1*i_z
                #print(a,names_increased_cut[a])
                name = names_increased_cut[a]
                each_mol = "mol%d = new %s.move(%4.3f,%4.3f,%4.3f).rot(%f,1,0,0,%4.3f,%4.3f,%4.3f).rot(%f,0,1,0,%4.3f,%4.3f,%4.3f)\n" % (a,name,x_pos,y_pos,z_pos,anglex1,x_pos,y_pos,z_pos,angley1,x_pos,y_pos,z_pos)
                strings+=each_mol
                a += 1

    strings+="#alt1\n"


    for i_x in range(0,x_num2,1):
        for i_y in range(1,y_num2,2):
            for i_z in range(z_num2):
                #print(i_x,i_y,i_z)
                x_pos = x_dist*i_x - (x_dist/y_num) * i_y
                y_pos = y_dist*i_y
                z_pos = z_dist1*i_z #+ 2
                #print(a,names_increased_cut[a])
                name = names_increased_cut[a]
                each_mol = "mol%d = new %s.move(%4.3f,%4.3f,%4.3f).rot(%f,1,0,0,%4.3f,%4.3f,%4.3f).rot(%f,0,1,0,%4.3f,%4.3f,%4.3f)\n" % (a,name,x_pos,y_pos,z_pos,anglex2,x_pos,y_pos,z_pos,angley2,x_pos,y_pos,z_pos)
                strings+=each_mol
                a += 1

    strings += '\n\nwrite_once("Data Boundary") {\n'
    strings += "0.0 %4.3f  xlo xhi\n" % (x_cell)
    strings += "0.0 %4.3f  ylo yhi\n" % (y_cell)
    strings += "0.0 %4.3f  zlo zhi\n" % (z_cell)
    strings += "0 0 0 xy xz yz\n"
    strings += "}\n"

    if not os.path.exists('./%s' % new_dir):
        os.mkdir('./%s' % new_dir)

    if not os.path.exists('./%s/moltemplate_files' % new_dir):
        os.mkdir('./%s/moltemplate_files' % new_dir)
    else:
        subprocess.call('rm -r ./%s/moltemplate_files/*.lt' % new_dir,shell=True)

    f = open('./%s/moltemplate_files/system.lt' % new_dir,'w')
    f.write(strings)
    f.close()

    print(x_cell,y_cell,z_cell)

    for mol_lt in mol_lt_files:
        subprocess.call('cp %s ./%s/moltemplate_files'% (mol_lt,new_dir),shell=True)