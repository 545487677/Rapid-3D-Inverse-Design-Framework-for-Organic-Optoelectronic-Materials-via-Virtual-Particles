

import os
from multiprocessing import Pool
from warnings import simplefilter
from tqdm import tqdm, trange
import sys
import re
import multiprocessing
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

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def rdkit_smi2mol(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
    except:
        return None
    return mol


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

def is_valid(smi, opt_mol=False):
        try:
            molecule = Chem.MolFromSmiles(smi)
            if mol2smiles(molecule) is None:
                return None
            else:
                bad_ring_smarts = {
                    'three_membered_ring': '[r3]',
                    'four_membered_ring': '[r4]',
                    'seven_membered_ring': '[r7]',
                    'eight_membered_ring': '[r8]'
                }
                queries = {name: Chem.MolFromSmarts(smarts) for name, smarts in bad_ring_smarts.items()}

                if any(molecule.HasSubstructMatch(query) for query in queries.values()):
                    print('Detect bad rings, skipping molecule.')
                    print(smi)
                    return None
                
                ssr = Chem.GetSymmSSSR(molecule)
                for i in range(len(ssr)):
                    for j in range(i+1, len(ssr)):
                                shared_atoms = set(ssr[i]).intersection(ssr[j])
                                if len(shared_atoms) > 1:
                                    ring1_size = len(ssr[i])
                                    ring2_size = len(ssr[j])
                                    if ring1_size < 5 or ring2_size < 5:
                                        print(f"Rings {i} and {j} are bridged and one of them is a small ring (size <= 5): {shared_atoms}")
                                        print(smi)
                                        return None
                if check_for_cube_structure(molecule):
                    print(smi)
                    return None

        except:
            return None
        molecule = Chem.AddHs(molecule)
        if opt_mol: ## Geometry optimization for molecules
            try:
                AllChem.EmbedMolecule(molecule)
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                return None
        return molecule



def convert_to_smiles(atoms_list, coords_list):
    smi_list = []
    for atoms, coords in zip(atoms_list, coords_list):
        try:
            smi = xyz2smi(atoms, coords)
            if '.' in smi:
                smi = None
            smi_list.append(smi)
        except Exception as e:
            # If conversion fails, log the error and append None
            logging.error(f"Error converting to SMILES: {e}")
            smi_list.append(None)
    return smi_list

# def is_mol_valid(mol):
#     # try:
#     # Chem.SanitizeMol(mol)
#     if mol is None:
#         return None,None
#     try:
#         frags = Chem.GetMolFrags(mol, asMols=True)
#         frags_atom = Chem.GetMolFrags(mol, asMols=False)
#         frags_len_max = max([len(x) for x in frags_atom])
#         for i in range(len(frags_atom)):
#             if len(frags_atom[i]) == frags_len_max:
#                 mol = frags[i]
#                 s = Chem.CanonSmiles(Chem.MolToSmiles(mol))
#                 return mol, s
#     except:
#         return None,None
#     return mol, s

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
    # smi = smi.replace('[H].', '').replace('.[H]', '')
    # smi = smi.replace('.[C]', '').replace('[C].', '')
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


# def convert_xyz_to_sdf(ligand_path:str, result_path:str):
#     temp_pdb_file = Path("../temp.pdb")
#     os.system("obabel {} -O {}".format(ligand_path, temp_pdb_file.as_posix()))
#     try:
#         mol = Chem.MolFromPDBFile(temp_pdb_file.as_posix(), sanitize=False)
#         mol = Chem.AddHs(mol)
#         repaired_mol = repair_one_mol(mol)
#         with Chem.SDWriter(result_path) as writer:
#             writer.write(repaired_mol)
#     except:
#         traceback.print_exc()

#     temp_pdb_file.unlink(missing_ok=True)

# def process_xyz_file(xyz_file):
#         # atoms, coords = extract_atoms_and_coords(xyz_file)
#         # smi = xyz2smi(atoms, coords)

#         sdf_filename2 = xyz_file[:-4]+'_temp.sdf'

#         convert_xyz_to_sdf(os.path.join(outpath, xyz_file), os.path.join(outpath,  sdf_filename2))

#         # # repair one mol
#         # try:
#         #     mol = Chem.MolFromSmiles(smi)
#         # except:
#         #     mol = None

#         if os.path.exists(os.path.join(outpath, sdf_filename2)):
#             try:
#                 mol = next(Chem.SDMolSupplier(os.path.join(outpath, sdf_filename2),removeHs=False))
#             except:
#                 mol = None
#         else:
#             mol = None
#         mol, smiles = is_mol_valid(mol)

#         if mol is not None: 
#             mol = Chem.AddHs(mol)
#             try:
#                 AllChem.EmbedMolecule(mol)
#                 AllChem.MMFFOptimizeMolecule(mol)
#                 print('MMFF done!')
#             except:
#                 pass

#         mol, smiles = is_mol_valid(mol)     
#         # print(smiles)

#         sdf_filename3 = xyz_file[:-4]+'_refine.sdf'

#         if smiles is not None:
#             try:
#                 os.remove(os.path.join(outpath, sdf_filename3))
#             except:
#                 pass
#             try:
#                 sdf_writer = Chem.SDWriter(os.path.join(outpath, sdf_filename3))
#                 sdf_writer.write(mol)
#                 sdf_writer.close()
#                 with open(os.path.join(outpath, xyz_file[:-4] + ".smi"),'w') as w:
#                     w.write(smiles) 
#             except:
#                 pass
#         else:
#             if os.path.exists(sdf_filename3):
#                 os.remove(sdf_filename3)

def process_xyz_file(file_path):
    atom_list = []
    coordinates_list = []
    condition_value = None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    atom_count_line = next((line for line in lines if 'V2000' in line or 'V3000' in line), None)
    if not atom_count_line:
        raise ValueError("Could not find atom count line with 'V2000' or 'V3000'")

    # atom_count = int(atom_count_line.split()[0])

    for line in lines[4:]: 
        parts = line.split()
        if len(parts) > 4:
            atom_list.append(parts[3])
            coordinates_list.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if len(parts) <= 4:  
            break        

    coordinates_array = np.array(coordinates_list)
    return condition_value, atom_list, coordinates_array, file_path.split('_')[-1].split('.sdf')[0]


# Define a function to format data for LMDB
def format_data_for_lmdb(index, atom_list, coordinates, file_index, smi):
    # Format the data as needed for LMDB
    data = pickle.dumps({'atoms': atom_list, 
                         'coordinates': [coordinates], # [coordinates.reshape(1, *coordinates.shape)]
                         'target': (0, 0),
                         'index': file_index, 'smi':smi}, protocol=-1)
    return index, data

def get_rank(file_path):
    try:
        file_name = os.path.basename(file_path)
        rank_number = int(re.search(r'rank_(\d+).sdf', file_name).group(1))
        return rank_number
    except (IndexError, ValueError, AttributeError):
        return float('inf')

if __name__ == '__main__':

    dir_list = [item for item in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, item))]
    file_list = []
    for item in dir_list:
        sdf_files = [x for x in os.listdir(os.path.join(inpath, item)) if x.endswith('.sdf')]
        file_list.extend([os.path.join(inpath, item, x) for x in sdf_files])

    file_list.sort(key=get_rank)
    print('file_list',len(file_list))

    with Pool(os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_xyz_file, file_list), total=len(file_list)))

    # Accumulators for all files
    all_conditions, all_atom_lists, all_coordinates, file_idx = zip(*results)

    smi_list = convert_to_smiles(all_atom_lists, all_coordinates)

    # Remove None values if any conversion failed
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        valid = list(tqdm(pool.imap(is_valid, smi_list), total=len(smi_list)))
    valid_id = [idx for idx, smile in enumerate(valid) if smile is not None]
    valid = [mol for mol in valid if mol is not None]
    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid]
    file_idx = [file_idx[idx] for idx in valid_id]
    all_coordinates = [all_coordinates[idx] for idx in valid_id]
    all_atom_lists = [all_atom_lists[idx] for idx in valid_id]
    smi_list = [smi_list[idx] for idx in valid_id]

    # Save SMILES list to a .npy file
    np.save(os.path.join(inpath,'smi_lists.npy'), valid_smiles)

    nthreads = multiprocessing.cpu_count()
    print("Number of CPU cores:", nthreads)

    outputfilename = os.path.join(inpath, "valid.lmdb")

    # Remove existing LMDB file if it exists
    try:
        os.remove(outputfilename)
    except FileNotFoundError:
        pass

    # Open a new LMDB environment
    env_new = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    txn_write = env_new.begin(write=True)

    # Prepare the data for multiprocessing
    params_list = [(i, all_atom_lists[i], all_coordinates[i], file_idx[i], smi_list[i]) for i in range(len(all_coordinates))]

    # Use Pool to process and write data to LMDB
    with Pool(os.cpu_count()) as pool:
        for index, lmdb_data in tqdm(pool.starmap(format_data_for_lmdb, params_list), total=len(params_list)):
            txn_write.put(f'{index}'.encode("ascii"), lmdb_data)
            if index % 10000 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)

    # Commit and close the LMDB environment
    txn_write.commit()
    env_new.close()
    print(f'Processed {len(params_list)} records into {outputfilename}')

    print('to lmdb done!')