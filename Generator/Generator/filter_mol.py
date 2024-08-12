

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
# import dpdata
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

simplefilter(action='ignore', category=FutureWarning)

inpath=sys.argv[1]
outpath = sys.argv[2]



def is_mol_valid(mol):
    # try:
    # Chem.SanitizeMol(mol)
    if mol is None:
        return None,None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
        frags_atom = Chem.GetMolFrags(mol, asMols=False)
        frags_len_max = max([len(x) for x in frags_atom])
        for i in range(len(frags_atom)):
            if len(frags_atom[i]) == frags_len_max:
                mol = frags[i]
                s = Chem.CanonSmiles(Chem.MolToSmiles(mol))
                return mol, s
    except:
        return None,None
    return mol, s

def extract_atoms_and_coords(xyz_filename):
    with open(xyz_filename, 'r') as file:
        lines = file.read().strip().split("\n")[2:]  # Skip the first two lines
        atoms = [line.split()[0] for line in lines]
        coords = [list(map(float, line.split()[1:])) for line in lines]
        return atoms, coords

# def visualize_generated_mol(mol, show_surface=False, opacity=0.5):
#     view = py3Dmol.view()

#     mblock = Chem.MolToMolBlock(mol)
#     view.addModel(mblock, 'mol')
#     view.setStyle({'model': -1}, {'stick': {}, 'sphere': {'radius': 0.35}})
#     if show_surface:
#         view.addSurface(py3Dmol.SAS, {'opacity': opacity}, {'model': -1})

#     view.zoomTo()
#     return view

def xyz2smi(atoms, coords):
    mol = openbabel.OBMol()
    for j in range(len(coords)):
        atom = mol.NewAtom()
        atom.SetAtomicNum(openbabel.GetAtomicNum(atoms[j]))
        x, y, z = map(float, coords[j])
        atom.SetVector(x, y, z)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    mol.AddHydrogens()
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat('smi')
    smi = obConversion.WriteString(mol)

    return smi.split('\t\n')[0]

# def set_property(mol: Chem.rdchem.Mol, key: str, value: Union[str, int, float]):
#     if isinstance(value, int):
#         mol.SetIntProp(key, value)
#     elif isinstance(value, float):
#         mol.SetDoubleProp(key, value)
#     elif isinstance(value, str):
#         mol.SetProp(key, value)

# def repair_one_mol(mol: Chem.rdchem.Mol, if_add_H=True) -> Chem.rdchem.Mol:
#     mol = dpdata.BondOrderSystem(rdkit_mol=mol, sanitize_level='high').get_mol()
#     mol = Chem.RemoveHs(mol)
#     if if_add_H:
#         the_properties_dict = mol.GetPropsAsDict()
#         mol_block = Chem.MolToMolBlock(mol, kekulize=True)
#         mol_str = subprocess.check_output(["obabel", "-imol", "-osdf", "-h"],
#                                           text=True, input=mol_block, stderr=subprocess.DEVNULL)
#         bstr = BytesIO(bytes(mol_str, encoding='utf-8'))
#         mol = next(Chem.ForwardSDMolSupplier(bstr, removeHs=False, sanitize=True))
#         for the_property in the_properties_dict.keys():
#             if the_property == "SMILES":
#                 removeHsMol = Chem.RemoveHs(mol)
#                 the_properties_dict[the_property] = Chem.MolToSmiles(removeHsMol)
#             set_property(mol=mol, key=the_property, value=the_properties_dict[the_property])
#     return mol


def convert_xyz_to_sdf(ligand_path:str, result_path:str):
    temp_pdb_file = Path("../temp.pdb")
    os.system("obabel {} -O {}".format(ligand_path, temp_pdb_file.as_posix()))
    try:
        mol = Chem.MolFromPDBFile(temp_pdb_file.as_posix(), sanitize=False)
        mol = Chem.AddHs(mol)
        # repaired_mol = repair_one_mol(mol)
        with Chem.SDWriter(result_path) as writer:
            # writer.write(repaired_mol)
            writer.write(mol)
    except:
        traceback.print_exc()

    temp_pdb_file.unlink(missing_ok=True)

def process_xyz_file(xyz_file):
        # atoms, coords = extract_atoms_and_coords(xyz_file)
        # smi = xyz2smi(atoms, coords)

        sdf_filename2 = xyz_file[:-4]+'_temp.sdf'

        convert_xyz_to_sdf(os.path.join(outpath, xyz_file), os.path.join(outpath,  sdf_filename2))

        # # repair one mol
        # try:
        #     mol = Chem.MolFromSmiles(smi)
        # except:
        #     mol = None

        if os.path.exists(os.path.join(outpath, sdf_filename2)):
            try:
                mol = next(Chem.SDMolSupplier(os.path.join(outpath, sdf_filename2),removeHs=False))
            except:
                mol = None
        else:
            mol = None
        mol, smiles = is_mol_valid(mol)

        if mol is not None: 
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                print('MMFF done!')
            except:
                pass

        mol, smiles = is_mol_valid(mol)     
        # print(smiles)

        sdf_filename3 = xyz_file[:-4]+'_refine.sdf'

        if smiles is not None:
            try:
                os.remove(os.path.join(outpath, sdf_filename3))
            except:
                pass
            try:
                sdf_writer = Chem.SDWriter(os.path.join(outpath, sdf_filename3))
                sdf_writer.write(mol)
                sdf_writer.close()
                with open(os.path.join(outpath, xyz_file[:-4] + ".smi"),'w') as w:
                    w.write(smiles) 
            except:
                pass
        else:
            if os.path.exists(sdf_filename3):
                os.remove(sdf_filename3)


if __name__ == '__main__':

    dir_list = [item  for item in os.listdir(inpath) if os.path.isdir(os.path.join(inpath,item)) ]
    file_list = []
    for item in dir_list:
        file_list.extend([os.path.join(inpath, item, x) for x in os.listdir(os.path.join(inpath, item)) if 'final_res' in x and x.endswith('.xyz') and 'loop0' not in x])
        if len([x for x in os.listdir(os.path.join(inpath, item)) if 'final_res' in x and x.endswith('.xyz') ])<504:
            print(item, len([x for x in os.listdir(os.path.join(inpath, item)) if 'final_res' in x and x.endswith('.xyz') ]))
    print('file_list',len(file_list))

    with Pool(os.cpu_count()) as pool:
        tqdm(list(pool.imap(process_xyz_file, file_list)), total=len(file_list))

    print('filter done!')
             
        # visualize_generated_mol(rd_mol)   
