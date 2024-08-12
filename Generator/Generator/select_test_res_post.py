

import os
from multiprocessing import Pool
from warnings import simplefilter
from tqdm import tqdm, trange
import sys
import logging
import subprocess
from io import BytesIO
from typing import Union
from pathlib import Path
import traceback
import glob
import lmdb
import pickle
import numpy as np
from openbabel import openbabel
# import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger
rdBase.DisableLog('rdApp.error')
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import networkx as nx
import shutil
from rdkit.Chem import rdchem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdqueries
from rdkit.Chem import rdFMCS
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
simplefilter(action='ignore', category=FutureWarning)


inpath=sys.argv[1]
outpath = sys.argv[2]
pdbid=sys.argv[3]
sample_size=sys.argv[4]

def check_for_cube_structure(mol):
    ssr = Chem.GetSymmSSSR(mol)
    
    atom_ring_count = {}

    for ring in ssr:
        for atom_idx in ring:
            if atom_idx in atom_ring_count:
                atom_ring_count[atom_idx] += 1
            else:
                atom_ring_count[atom_idx] = 1

    potential_cube_vertices = [atom_idx for atom_idx, count in atom_ring_count.items() if count >= 3]

    if len(potential_cube_vertices) >= 4:
        print("Molecule may contain a cubic structure.")
        return True
    else:
        return False

def remove_chirality(mol):
    if mol is None:
        raise ValueError("Invalid SMILES string")
    for atom in mol.GetAtoms():
        atom.SetChiralTag(rdchem.ChiralType.CHI_UNSPECIFIED)
    return mol

def mw_500(m):
    mw = np.round(Descriptors.MolWt(m),1)
    return mw < 500

def hba_10(mol):
    if not mol:
        return False
    smarts="[$([n;H0;X2]),$([#8;H0,H1]),$(N(=A)-A),$(N#C)]"
    pattern = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(pattern)
    return len(matches) < 10

def hbd_5(mol):
    if mol is None:
        return False
    smarts = "[$([#8,#7,#16;!H0])]"
    pattern = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(pattern)
    return len(matches) < 5

def ring_size9(mol):
    max_ring_size = 9
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) > max_ring_size:
            return False
    return True

def pains_rule(mol):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains = FilterCatalog.FilterCatalog(params)
    alerts = pains.GetMatches(mol)
    return len(alerts) == 0

def qed_04(mol):
    qed = np.round(QED.qed(mol),2)
    return qed > 0.4

def get_qed(mol):
    qed = np.round(QED.qed(mol),2)
    return qed

def combine_lst(l):
    G = nx.Graph()
    G.add_nodes_from(sum(l, []))
    q = [[(s[i], s[i + 1]) for i in range(len(s) - 1)] for s in l]
    for i in q:
        G.add_edges_from(i)
    return [list(i) for i in nx.connected_components(G)]

def get_ring_atom(mol):

    ring_atoms = []
    non_ring_atoms = []
    all_atoms = list(range(mol.GetNumAtoms()))
    ssr = Chem.GetSymmSSSR(mol)
    atom_dict = {}
    i=0
    for ring in ssr:
        ring_atoms.append(list(ring))
        for atom in list(ring):
            if atom not in atom_dict:
                atom_dict[atom] = [i]
            else:
                atom_dict[atom].append(i)
        i+=1
    ring_atoms = combine_lst(ring_atoms)

    ring_num = []
    for ring in ring_atoms:
        ring_num_t = []
        for atom in ring:
            ring_num_t.extend(atom_dict[atom])
        ring_num.append(len(set(ring_num_t)))
    
    return ring_atoms, ring_num


def select(item):
    pdbid = item
    subdirs = [os.path.join(inpath, d) for d in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, d))]
    three_membered_ring_smarts = '[r3]'
    four_membered_ring_smarts = '[r4]'
    seven_membered_ring_smarts = '[r7]'
    eight_membered_ring_smarts = '[r8]'    
        
    three_membered_ring_query = Chem.MolFromSmarts(three_membered_ring_smarts)
    four_membered_ring_query = Chem.MolFromSmarts(four_membered_ring_smarts)
    seven_membered_ring_query = Chem.MolFromSmarts(seven_membered_ring_smarts)
    eight_membered_ring_query = Chem.MolFromSmarts(eight_membered_ring_smarts)


    file_list = [os.path.join(subdir, f) for subdir in subdirs for f in os.listdir(subdir) if f.endswith('_refine.sdf')]    
    score_dict = {}
    selected_dict = {}
    for item2 in file_list:
        sdf_name = os.path.join(inpath, item, item2)
       ## If the coordinates of any axis are all 0.0000, the file will not be written.
        with open(sdf_name, 'r') as file:
            lines = file.readlines()

        atoms = []
        for line in lines[4:]:
            parts = line.split()
            if len(parts) < 4:
                continue 
            if not parts[3].isalpha():
                continue 
            atoms.append(parts)

        x_zero, y_zero, z_zero = True, True, True
        for parts in atoms:
            x, y, z = parts[0:3]
            if x != '0.0000': x_zero = False
            if y != '0.0000': y_zero = False
            if z != '0.0000': z_zero = False

        if x_zero or y_zero or z_zero:
            print(f"warning: the coordinates of any axis are all 0.0000")
            continue



        xyz_name = os.path.join(inpath, item, item2[:-11]+'.xyz')
        try:
            suppl = Chem.SDMolSupplier(sdf_name)
            mol = [mol for mol in suppl if mol][0]
        except:
            continue
        xyz_head = open(xyz_name,'r').readlines()[1]
        plddt = float(xyz_head.split('|')[-1].split(':')[-1])
        pred_atom_num = float(xyz_head.split('|')[5].split(':')[-1])
        # used_atom_num =  int(xyz_head.split('|')[6].split(':')[-1])
        all_mol_atom = int(open(xyz_name,'r').readlines()[0].strip())
        mol_atom_num = mol.GetNumAtoms()
        basename = os.path.basename(item2)  # '69_final_res16_1_0_1_98204_loop1_refine.sdf'

        name_without_ext = os.path.splitext(basename)[0][:-7]  # '69_final_res16_1_0_1_98204_loop1'

        new_name = name_without_ext + '.smi'  # '69_final_res16_1_0_1_98204_loop1.smi'

        new_path = os.path.join(os.path.dirname(item2), new_name)

        smi = open(os.path.join(inpath, item, new_path),'r').read().strip()

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
 
        if smi in selected_dict:
            continue
        else:
            selected_dict[smi] = 1

        if float(mol_atom_num)/all_mol_atom < 1:
            if 'optimization' in outpath:
                score_dict[item2] = min(float(mol_atom_num)/all_mol_atom, plddt) - 0.02 #-1 + float(mol_atom_num)/all_mol_atom - 1
            else:
                score_dict[item2] = min(float(mol_atom_num)/all_mol_atom, plddt)
        else:
            score_dict[item2] = plddt #+ abs(pred_atom_num - all_mol_atom) * 0.002
        
        if score_dict[item2] <  0.935:
            score_dict[item2] = -100


        mol = remove_chirality(mol)
        if (mol.HasSubstructMatch(three_membered_ring_query) or 
            mol.HasSubstructMatch(four_membered_ring_query) or
            mol.HasSubstructMatch(seven_membered_ring_query) or
            mol.HasSubstructMatch(eight_membered_ring_query)):
            print('Detect bad rings, skipping molecule.')
            continue
            
            
        ssr = Chem.GetSymmSSSR(mol)

        has_three_membered_ring = any(len(ring) == 3 for ring in ssr)
        if has_three_membered_ring:
            print('detect three rings')
            continue

        has_four_membered_ring = any(len(ring) == 4 for ring in ssr)
        if has_four_membered_ring:
            print('detect four rings')
            continue    

        for i in range(len(ssr)):
            for j in range(i+1, len(ssr)):
                        shared_atoms = set(ssr[i]).intersection(ssr[j])
                        if len(shared_atoms) > 1:
                            ring1_size = len(ssr[i])
                            ring2_size = len(ssr[j])
                            
                            if ring1_size < 5 or ring2_size < 5:
                                print(f"Rings {i} and {j} are bridged and one of them is a small ring (size <= 5): {shared_atoms}")
                                # break
                                continue
                                # score_dict[item2] = score_dict[item2] - 9999
                                

        if check_for_cube_structure(mol):
            # break
            continue
            # score_dict[item2] = score_dict[item2] - 9999



        # qed_score = get_qed(mol)
        # if qed_score < 0.4:
        #     score_dict[item2] = min(score_dict[item2], qed_score)
        if not (mw_500(mol) & hba_10(mol) & hbd_5(mol) & ring_size9(mol) & pains_rule(mol)):
            score_dict[item2] = score_dict[item2] -1.1

    score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    json_dict=[]
    index = 0
    for item in score_dict:
        if  index > int(sample_size) and item[1] < 0:
            break
        # if index > 5 and item[1] < 0.935:
        #     break
        # if index > 5 and item[1] < -10:
        #     break
        # os.system('cp ' + os.path.join(inpath, item[0].split('_')[0], item[0] ) + ' ' + os.path.join(outpath, item[0].split('_')[0], item[0] ))
        os.system('cp ' + os.path.join(inpath, pdbid, item[0] ) + ' ' + os.path.join(outpath, pdbid, 'rank_'+str(index)+'.sdf' ))
        
        with open(os.path.join(outpath, pdbid, 'rank_'+str(index)+'_res.txt'), 'w') as w:
            w.write(str(item[1] )+'\n')
            w.write(item[0]+'\n')
        json_dict.append({
                'ligand_path': f'rank_{index}.sdf', #relpath
                'rank_order': str(index),
                'energy': item[1],
                'name': item[0],
            })
        index+=1
    import json
    json_data = json.dumps(json_dict, indent=4)
    with open(os.path.join(outpath,'result.vdgen.json'), 'w') as f:
        f.write(json_data)
if __name__ == '__main__':
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    else:
        # os.remove(outpath)
        shutil.rmtree(outpath)
        os.mkdir(outpath)

    if not os.path.exists(os.path.join(outpath, pdbid)):
        os.mkdir(os.path.join(outpath, pdbid))
    
    select(pdbid)
    print("select done!")
