#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd 
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import itertools
import pickle
import lmdb
import multiprocessing
from multiprocessing import Pool

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")

RANGE_DICT = {
    'homo': (-6.754576, -4.587008),
    'lumo': (-4.4918088, -0.342176),
 
}

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

def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)

# def is_valid_smiles(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     return mol2smiles(mol) is not None

def is_valid_smiles(smi, opt_mol=False):
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
        return mol2smiles(molecule) is not None

def filter_smiles(df):
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        smiles_iter = df['SMILES']
        results = list(tqdm(pool.imap(is_valid_smiles, smiles_iter), total=len(smiles_iter)))
        valid_indices = [i for i, valid in enumerate(results) if valid]
    return df.iloc[valid_indices].reset_index(drop=True)


def generate_xyz_content(atoms, coords):
    content = f"{len(atoms)}\n\n"
    for atom, coord in zip(atoms, coords):
        content += f"{atom} {' '.join(map(str, coord))}\n"
    return content

def generate_xyz_and_compress(results_path, atoms_list, coords_list, filtered_pred_df, to_lmdb=None):
    if not to_lmdb:
        outpath = os.path.join(results_path, 'xyz')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        else:
            shutil.rmtree(outpath)
            os.mkdir(outpath)
    atoms_list = np.array(atoms_list, dtype=object)
    coords_list = np.array(coords_list, dtype=object)

    if to_lmdb:
        outputfilename = os.path.join(results_path, 'db.lmdb')
        env = lmdb.open(outputfilename,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(10e9))

    for idx, row in filtered_pred_df.iterrows():
        index = row['Index']
        atoms = atoms_list[index]
        coords = coords_list[index]

        assert len(atoms) == len(coords), "Atoms and coordinates count does not match."
        
        if to_lmdb:
            with env.begin(write=True) as txn:
                smiles = row['SMILES'].encode()
                if txn.get(smiles) is None:
                    data_dict = row.to_dict()
                    data_dict['atoms'] = atoms
                    data_dict['coords'] = coords
                    serialized_data = pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL)
                    txn.put(smiles, serialized_data)
        else:
            xyz_file_path = os.path.join(outpath, f'molecule_{index}.xyz')
            with open(xyz_file_path, 'w') as file:
                file.write(generate_xyz_content(atoms, coords))

    if not to_lmdb:
        print("XYZ files have been successfully saved.")
        archive_path = shutil.make_archive(base_name=outpath, format='zip', root_dir=outpath)
        print(f"Created archive at: {archive_path}")
        shutil.rmtree(outpath)
        print(f"Deleted original xyz files at: {outpath}")
    else:
        print("Data has been stored to LMDB.")

def filter_by_gap(filtered_pred_df: pd.DataFrame, gap_type: str):
    filtered_pred_df["gap"] = filtered_pred_df["lumo"] - filtered_pred_df["homo"]
    mean_gap = filtered_pred_df["gap"].mean()
    std_gap = filtered_pred_df["gap"].std()

    if gap_type == 'max':
        threshold = mean_gap + 1 * std_gap
        return filtered_pred_df[filtered_pred_df["gap"] > threshold]
    elif gap_type == 'min':
        threshold = mean_gap - 1 * std_gap
        return filtered_pred_df[filtered_pred_df["gap"] < threshold]
    else:
        print("Error: gap_type must be 'max' or 'min'.")
        return filtered_pred_df  


def voc_cal(Dhomo, Alumo):
    return Alumo - Dhomo - 0.3

def dE_E_cal(Alumo, Dlumo):
    return Alumo - Dlumo

def filter_and_extract_top_pairs(filtered_pred_df, voc_cal, dE_E_cal, results_path):

    homo_array = filtered_pred_df['homo'].values
    lumo_array = filtered_pred_df['lumo'].values
    Dhomo = homo_array[:, np.newaxis]
    Alumo = lumo_array[np.newaxis, :]
    voc_matrix = voc_cal(Dhomo, Alumo)
    de_e_matrix = dE_E_cal(Alumo, lumo_array[:, np.newaxis])

    num_rows, num_cols = voc_matrix.shape
    flat_voc_values = voc_matrix.ravel()
    original_indices = np.arange(num_rows*num_cols)
    row_indices, col_indices = np.divmod(original_indices, num_cols)
    N = int(filtered_pred_df.shape[0] * 0.60)
    print(N)
    top_100_voc_indices = np.argsort(flat_voc_values)[-N:]
    top_100_voc_values = flat_voc_values[top_100_voc_indices]
    top_100_voc_row_indices = row_indices[top_100_voc_indices]
    top_100_voc_col_indices = col_indices[top_100_voc_indices]

    flat_de_e_values = de_e_matrix.ravel()
    top_100_de_e_indices = np.argsort(flat_de_e_values)[-N:]
    top_100_de_e_values = flat_de_e_values[top_100_de_e_indices]
    top_100_de_e_row_indices = row_indices[top_100_de_e_indices]
    top_100_de_e_col_indices = col_indices[top_100_de_e_indices]

    top_voc_100_pairs_df = pd.DataFrame({
        'Donor_SMILES': filtered_pred_df.iloc[top_100_voc_row_indices]['SMILES'].values,
        'Acceptor_SMILES': filtered_pred_df.iloc[top_100_voc_col_indices]['SMILES'].values,
        'Voc_Value': top_100_voc_values
    })

    top_de_e_100_pairs_df = pd.DataFrame({
        'Donor_SMILES': filtered_pred_df.iloc[top_100_de_e_row_indices]['SMILES'].values,
        'Acceptor_SMILES': filtered_pred_df.iloc[top_100_de_e_col_indices]['SMILES'].values,
        'dE_E_Value': top_100_de_e_values
    })

    all_smiles_set = set(top_voc_100_pairs_df['Donor_SMILES']) | set(top_voc_100_pairs_df['Acceptor_SMILES']) | set(top_de_e_100_pairs_df['Donor_SMILES']) | set(top_de_e_100_pairs_df['Acceptor_SMILES'])
    filtered_pred_df = filtered_pred_df[filtered_pred_df['SMILES'].isin(all_smiles_set)]
    
    filtered_pred_df.to_csv(results_path, index=False)
    return filtered_pred_df




def calculate_r2_scores_for_each_column(target_array, prediction_array):

    r2_scores = {}
    for column_index in range(target_array.shape[1]):
        r2_score_for_column = r2_score(target_array[:, column_index], prediction_array[:, column_index])
        r2_scores[f"Column {column_index}"] = r2_score_for_column
    return r2_scores

def calculate_mae_scores_for_each_column(target_array, pred_array):
    mae_scores = {}
    for column in range(target_array.shape[1]):
        mae_scores[column] = mean_absolute_error(target_array[:, column], pred_array[:, column])
    return mae_scores

def create_special_mask(src_tokens, special_idx_tensor):
    return torch.isin(src_tokens, special_idx_tensor.to(src_tokens))







def main(args):

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    # state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    # model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_cuda:
        model.cuda()
        # fp16 only supported on CUDA for fused kernels
        if use_fp16:
            model.half()

    # Print args
    logger.info(args)

    # Build loss
    loss = task.build_loss(args)
    loss.eval()

    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        try:
            fname = (args.path).split("/")[-2]
        except:
            fname = 'infer'

        if not os.path.exists(os.path.join(args.results_path, 'xyz')):
            os.makedirs(os.path.join(args.results_path, 'xyz'))
        try:
            fname = (args.path).split("/")[-2]
        except:
            fname = 'xyz'
        # save_path = os.path.join(args.results_path, fname + "_" + subset + ".out.pkl")
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        log_outputs = []
        pred_list = []
        smi_list = []
        coords_list = []
        atoms_list = []
        target_list = []
        indices = {v:k for k,v in task.dictionary.indices.items()}
        special_idx_tensor = torch.tensor(task.dictionary.special_index())

        # index2atoms = task.dictionary
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            _, _, log_output = task.valid_step(sample, model, loss, test=True)

            if args.todft:
                mask = (~(create_special_mask(sample["ori_tokens"], special_idx_tensor))).int()

                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(sample['ori_coords'])

                for token, coords in zip(sample["ori_tokens"] * mask, sample['ori_coords'] * mask_expanded):
                    non_zero_tokens = token[token!=0].cpu().numpy()
                    atoms = [indices[i] for i in non_zero_tokens]
                    atoms_list.append(atoms)
                    
                    non_zero_coords = coords[(coords != 0).any(dim=1)].cpu().numpy()
                    coords_list.append(non_zero_coords)
                    
                    assert len(atoms) == len(non_zero_coords), "The number of atoms does not match the number of coordinates"

            pred_list.extend(log_output["predict"].cpu().detach().numpy())
            smi_list.extend(sample['smi_name'])
            progress.log({}, step=i)
            log_outputs.append(log_output)
            target_list.extend(log_output['target'].cpu().detach().numpy()) 

        pred_list = np.array(pred_list)
        target_list = np.array(target_list)

        if not np.all(target_list == 0):
            r2_scores = calculate_r2_scores_for_each_column(target_list, pred_list)

            for column, r2_score in r2_scores.items():
                print(f'R2 for {column}: {r2_score}')

            mae_scores = calculate_mae_scores_for_each_column(target_list, pred_list)

            for column, mae_score in mae_scores.items():
                print(f'MAE for column {column}: {mae_score}')        

        pred_df = pd.DataFrame(pred_list, columns=RANGE_DICT.keys())
        if len(smi_list) == len(pred_df):
            pred_df.insert(0, 'SMILES', smi_list)
        else:
            print("warning: SMILES list length does not match the DataFrame row count")

        filtered_pred_df = pred_df.copy()

        for col, (min_val, max_val) in RANGE_DICT.items():
            if col in filtered_pred_df.columns:
                filtered_pred_df = filtered_pred_df[(filtered_pred_df[col] >= min_val) & (filtered_pred_df[col] <= max_val)]
        filtered_pred_df.insert(0, 'Index', filtered_pred_df.index)

        filtered_pred_df = filter_smiles(filtered_pred_df)  
        filtered_pred_df.to_csv(os.path.join(args.results_path, 'final_res.csv'), index=False)
        print(filtered_pred_df.shape)

        if hasattr(args, 'gap_type') and args.gap_type in ['max', 'min']:
            filtered_pred_df = filter_by_gap(filtered_pred_df, args.gap_type)
            print(filtered_pred_df.shape)
            filtered_pred_df.to_csv(os.path.join(args.results_path, f'final_res_screening_gap_with_{args.gap_type}.csv'), index=False)


        if args.strict_filter:
            filtered_pred_df = filter_and_extract_top_pairs(filtered_pred_df, voc_cal, dE_E_cal, os.path.join(args.results_path, 'final_strict_screen_res.csv'))
            print(filtered_pred_df.shape)
            filtered_pred_df.to_csv(os.path.join(args.results_path, f'final_res_screening_with_strict_filter.csv'), index=False)

        ## to xyz / to lmdb
        if args.todft:
            generate_xyz_and_compress(args.results_path, atoms_list, coords_list, filtered_pred_df, to_lmdb=args.tolmdb)


        logger.info("Done inference! ")

    return None


def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
