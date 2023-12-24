import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase.io import read, write
import numpy as np
import numpy.linalg as LA
import os, sys, glob, subprocess
from itertools import combinations
from copy import deepcopy
import itertools
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.mplot3d import Axes3D

### Minimum bounding box 3D
def yaw(theta):
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]])

def pitch(theta):
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta) , 0, np.sin(theta)],
                     [             0, 1,             0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def roll(theta):
    theta = np.deg2rad(theta)
    return np.array([[1,              0,             0],
                     [0,  np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])


def CreateFigure():
    """
    Helper function to create figures and label axes
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax


def draw3DRectangle(ax, x1, y1, z1, x2, y2, z2):
    # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
    ax.plot([x1, x2], [y1, y1], [z1, z1], color='b') # | (up)
    ax.plot([x2, x2], [y1, y2], [z1, z1], color='b') # -->
    ax.plot([x2, x1], [y2, y2], [z1, z1], color='b') # | (down)
    ax.plot([x1, x1], [y2, y1], [z1, z1], color='b') # <--

    ax.plot([x1, x2], [y1, y1], [z2, z2], color='b') # | (up)
    ax.plot([x2, x2], [y1, y2], [z2, z2], color='b') # -->
    ax.plot([x2, x1], [y2, y2], [z2, z2], color='b') # | (down)
    ax.plot([x1, x1], [y2, y1], [z2, z2], color='b') # <--
    
    ax.plot([x1, x1], [y1, y1], [z1, z2], color='b') # | (up)
    ax.plot([x2, x2], [y2, y2], [z1, z2], color='b') # -->
    ax.plot([x1, x1], [y2, y2], [z1, z2], color='b') # | (down)
    ax.plot([x2, x2], [y1, y1], [z1, z2], color='b') # <--

def Minimum_Bounding_Box_3D(positions,center_at_origin=True):
    data = np.vstack([positions[:,0], positions[:,1], positions[:,2]])
    means = np.mean(data, axis=1)
    cov = np.cov(data)
    eval, evec = LA.eig(cov)
    centered_data = data - means[:,np.newaxis]
    aligned_coords = np.matmul(evec.T, centered_data)
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])
    # Realign with original coordinats
    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]
    
    box_info = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    x_diff = xmax - xmin
    y_diff = ymax - ymin
    z_diff = zmax - zmin
    ranking = sorted([[0,x_diff],[1,y_diff],[2,z_diff]],key=lambda x:float(x[1]))
    new_z = ranking[0][0]
    new_x = ranking[1][0]
    new_y = ranking[2][0]
    new_box_info = [box_info[new_x],box_info[new_y],box_info[new_z]]
    new_box_info_shift = [[0,new_box_info[0][1]-new_box_info[0][0]],[0,new_box_info[1][1]-new_box_info[1][0]],[0,new_box_info[2][1]-new_box_info[2][0]]]
    if center_at_origin:
        new_positions = np.array([[x,y,z] for x,y,z in zip(aligned_coords[new_x],aligned_coords[new_y],aligned_coords[new_z])])
    else:
        new_positions = np.array([[x-new_box_info[0][0],y-new_box_info[1][0],z-new_box_info[2][0]] for x,y,z in zip(aligned_coords[new_x],aligned_coords[new_y],aligned_coords[new_z])])
    
    # Realign with original coordinats
    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]
    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                      [y1, y2, y2, y1, y1, y2, y2, y1],
                                                      [z1, z1, z1, z1, z2, z2, z2, z2]])
    # rrc = rotated rectangle coordinates
    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    rrc += means[:, np.newaxis] 
    #print(rrc)
    return new_positions, new_box_info_shift

def ReadPositionInPDBfile(pdb_string):
    #f = open(filename,'r')
    lines = pdb_string.split('\n')
    positions = np.array([line.split()[5:8] for line in lines if 'HETATM' in line.split() or 'ATOM' in line.split() ])
    positions = positions.astype(float)
    return positions

def UpdatePositionInPDBfile(pdb_string, positions, box_info):
    input_pdb_string = pdb_string
    lines = input_pdb_string.split('\n')

    pdb_string = ''
    for i, line in enumerate(lines):
        if 'END' in line:
            pdb_string += 'END #3DBOX %4.3f %4.3f %4.3f' % (box_info[0][1],box_info[1][1],box_info[2][1])
        elif 'HETATM' in line:
            lt = line.split()
            lt[5] = "%5.3f" % positions[i][0]
            lt[6] = "%5.3f" % positions[i][1]
            lt[7] = "%5.3f" % positions[i][2]
            new_lt = "%s%5s%4s%5s%6s%12s%8s%8s%6s%6s%12s\n" % (lt[0],lt[1],lt[2],lt[3],lt[4],lt[5],lt[6],lt[7],lt[8],lt[9],lt[10])
            pdb_string+=new_lt
        else:
            pdb_string+=line + '\n'
    return pdb_string

def Get3DMinimumBoundingBox(pdb_string,format='pdb'):

    if format == 'pdb':
        positions = ReadPositionInPDBfile(pdb_string)
        new_positions, new_box_info = Minimum_Bounding_Box_3D(positions)
        new_pdb_string = UpdatePositionInPDBfile(pdb_string, new_positions, new_box_info)

    return new_box_info, new_pdb_string