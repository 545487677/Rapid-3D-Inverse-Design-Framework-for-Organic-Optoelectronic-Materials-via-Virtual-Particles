
import numpy as np
from rdkit import Chem
import openbabel
import os
import glob
import random
import re
import multiprocessing
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger
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

def canonical_smiles(smi):
    try:
        RDLogger.DisableLog('rdApp.*')
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Chem.MolSanitizeException:
        return None

def ob_xyz2smi(atoms, coords):
    mol = openbabel.OBMol()
    for j in range(len(coords)):
        atom = mol.NewAtom()
        atom.SetAtomicNum(openbabel.GetAtomicNum(atoms[j]))
        x, y, z = map(float, coords[j])
        atom.SetVector(x, y, z)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat('smi')
    smi = obConversion.WriteString(mol)
    return smi.split('\t\n')[0]

def read_xyz_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    atom_count = int(lines[0].strip())
    atoms = []
    coordinates = []
    for line in lines[2:2 + atom_count]:
        parts = line.split()
        atom_type, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(atom_type)
        coordinates.append((x, y, z))
    return atoms, np.array(coordinates)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def get_valid_smiles_from_xyz(file_path):
    atoms, coordinates_array = read_xyz_file(file_path)
    smi = ob_xyz2smi(atoms, coordinates_array)
    mol = rdkit_smi2mol(smi)
    is_connect = True
    if mol is None:
        return None, False, 0
    else: 
        smi = mol2smiles(mol)
        mol = rdkit_smi2mol(smi)
        
        if smi is not None:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            if len(mol_frags) > 1:
                is_connect = False
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            num_rings = len(Chem.GetSSSR(largest_mol))
            return mol2smiles(largest_mol), is_connect, num_rings
    return None, False, 0

def calculate_metrics(path, sample_length, train_data_path, seed):
    invalid_smi_files = []
    random.seed(seed)
    np.random.seed(seed)
    connected_count = 0 
    smi_list = []
    total_rings = 0

    subdir = [os.path.join(path, i) for i in os.listdir(path)]
    subdir = sorted(subdir, key=natural_sort_key)
    sdf_files = []
    for folder in subdir:
        search_pattern = os.path.join(folder, '*_loop1.xyz')
        sdf_files.extend(glob.glob(search_pattern))
    print(len(sdf_files))
    sdf_files = sdf_files[:sample_length]
    for file in sdf_files:
        smi, is_connected, num_rings = get_valid_smiles_from_xyz(file)
        if smi is None:
            invalid_smi_files.append(file)
        else:
            smi_list.append(smi)
            if is_connected:
                connected_count += 1
            total_rings += num_rings
    valid_smi_list = [smi for smi in smi_list if smi is not None]
    valid_count = len(valid_smi_list)
    valid_ratio = valid_count / sample_length
    unique_smiles = set(valid_smi_list)
    unique_count = len(unique_smiles)
    unique_ratio = unique_count / valid_count
    connected_ratio = connected_count / sample_length

    average_rings_per_molecule = total_rings / valid_count if valid_count else 0

    novelty_ratio = 1.0
    # Novelty calculation
    # train_data = train_data = np.load(train_data_path,allow_pickle=True)
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         gen_smiles = list(tqdm(pool.imap(canonical_smiles, valid_smi_list), total=len(valid_smi_list)))

    #         gen_smiles = [smi for smi in gen_smiles if smi is not None]
                
    #         train_smiles = [smi for smi in train_data if smi is not None]
    #         if len(gen_smiles) == 0:
    #                 novelty_ratio = 0.
    #         else:
    #                 duplicates = [1 for mol in gen_smiles if mol in train_smiles]
    #                 novel = len(gen_smiles) - sum(duplicates)
    #                 novelty_ratio = novel/len(gen_smiles)  
                    
    return valid_ratio, unique_ratio, novelty_ratio, connected_ratio, average_rings_per_molecule, valid_smi_list, invalid_smi_files

path = '../scripts/compress_1.0_10000_with_pretrain_2_grid_size_0.1_grid_offset_size_seed_1_1/res_optim'
sample_length = 10000
train_data_path = '../OPV_DATA_4_BENCH/train_canonical_smiles.npy' 
seed = 42
valid_ratio, unique_ratio, novelty_ratio, connected_ratio, average_rings_per_molecule, valid_smi_list, invalid_smi_files = calculate_metrics(path, sample_length, train_data_path, seed)
print(f"Valid ratio: {valid_ratio}, Unique ratio: {unique_ratio}, Novelty ratio: {novelty_ratio}, Connected_ratio: {connected_ratio}, average_rings_per_molecule:{average_rings_per_molecule}")