#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import os
import sys
import pickle
import numpy as np
from argparse import Namespace
from itertools import chain, islice

import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import metrics, progress_bar
from unicore import tasks


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unet3d.inference")


def main(args):
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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


    # Load ensemble
    if getattr(args, "finetune_encoder_model", None) is not None or getattr(args, "pretrain_path", None) is not None:
        logger.info("loading model(s) from {}".format(args.pretrain_path))
        state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrain_path)
    else:
        logger.info("loading model(s) from {}".format(args.path))
        state = checkpoint_utils.load_checkpoint_to_cpu(args.path)        
    
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

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

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            # required_batch_size_multiple=args.required_batch_size_multiple,
            required_batch_size_multiple=1,
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
        path_name = args.results_path
        # if 'unimol' in args.task:
        target_key = 'target'
        log_outputs = []
        if args.calculate_metric > 0:
            outpath = os.path.join('/'.join(args.path.split('/')[:-1]),path_name)
        else:
            outpath = os.path.join(args.results_path, args.results_name)
        if not os.path.exists(outpath):
            try:
                os.mkdir(outpath)
            except:
                pass
        repeated_num = args.method_num
        tt = 0
        start_sample_length = 0
        ## embedding 
        smi_list_0 = []
        embedding_list_loop0_0 = []
        smi_list = []
        embedding_list = []

        ##
        for j, sample in enumerate(progress):
                
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            
            _loss, _sample_size, log_output, all_output = task.inference_step(sample, model, loss, j)
            if start_sample_length >= int(args.sample_length):
                print('sample :', start_sample_length )
                print('finished!')
                break
            start_sample_length += log_output['bsz']
            print('sample :', start_sample_length )
            smis = sample['net_input']['smi']
            mol_idxs = sample['net_input']['mol_idx']


            unmask_index = sample[target_key]['unmask_index']


            all_atoms = sample[target_key]['all_atom'] 
            all_coords = sample[target_key]['all_coord']

            if args.grid_vis:
                init_grid = sample[target_key]['grid_coords']
                init_grid_noise_atom_coords = sample[target_key]['grid_noise_atom_coords']
            for loop_num, output in enumerate(all_output):

                if loop_num == 0:
                    tt = 0
                    kk = 0
                    mol_idxs = mol_idxs.unsqueeze(1).repeat(1,repeated_num).reshape(-1)
                    smis = []
                    for item in sample['net_input']['smi']:
                        for _ in range(repeated_num):
                            smis.append(item)
                    pdbids = []
                    for item in sample['net_input']['mol_idx']:
                        item = str(int(item.cpu().data))
                        for _ in range(repeated_num):
                            pdbids.append(item)
                    
                    
                    unmask_index = unmask_index.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,unmask_index.size(1))

                    all_atoms = all_atoms.repeat(1,repeated_num,1).reshape(-1,all_atoms.size(1))
                    all_coords = all_coords.repeat(1,repeated_num,1,1).reshape(-1,all_coords.size(1), all_coords.size(2))

                if loop_num == 1:
                    tt = 0
                    kk = 0
                    mol_idxs = mol_idxs.unsqueeze(1).repeat(1,repeated_num).reshape(-1)
                    smis = []
                    for item in sample['net_input']['smi']:
                        for _ in range(repeated_num):
                            smis.append(item)
                    pdbids = []
                    for item in sample['net_input']['mol_idx']:
                        item = str(int(item.cpu().data))
                        # assert 1==0, item
                        for _ in range(repeated_num):
                            pdbids.append(item)
                    
                    
                    unmask_index = unmask_index.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,unmask_index.size(1))

                    all_atoms = all_atoms.repeat(1,repeated_num,1).reshape(-1,all_atoms.size(1))
                    all_coords = all_coords.repeat(1,repeated_num,1,1).reshape(-1,all_coords.size(1), all_coords.size(2))

  
                if loop_num == 0:
                    logits_encoder, encoder_coord, all_atoms, all_coords, virtual_index, all_atoms2, used_atom_num, used_auc, add_atom_num, merge_method, pred_lddt, src_tokens, masked_coord, encoder_rep = output               
                else:
                    logits_encoder, encoder_coord, all_atoms, all_coords, virtual_index, all_atoms2, used_atom_num, used_auc, add_atom_num, merge_method,  src_tokens, masked_coord, encoder_rep, pred_lddt  = output                             
                
                if loop_num == 0:

                    embedding_list_loop0_0.append(np.mean(encoder_rep.cpu().numpy(), axis=1))

                if loop_num == 1:
                    smi_list.append(np.array(sample['net_input']['smi']))
                    embedding_list.append(np.mean(encoder_rep.cpu().numpy(), axis=1))
 
                if loop_num==0:
                    virtual_index = sample[target_key]['virtual_index']

                pred_atoms = logits_encoder.argmax(dim=-1)
                pred_coords = encoder_coord 
                
                for i in range(all_atoms.size(0)):
                    smi = smis[i]
                    mol_idx = mol_idxs[i].item()
                    if loop_num ==0:
                        virtual_num  = all_atoms[i].eq(model.mask_idx).sum().item()
                        assert virtual_num == virtual_index[i].sum().item(), (virtual_num, virtual_index[i].sum().item(), loop_num)
                    else:
                        virtual_num = virtual_index[i].sum().item()
                    padding_ = all_atoms[i][1:].ne(task.dictionary.pad())
                    all_atom = all_atoms[i][1:][padding_]
                    all_coord = all_coords[i][1:][padding_,:]
                    
                    atom_types = []
                    atom_x = []
                    atom_y = []
                    atom_z = []
                    init_atom_types = []
                    init_atom_x = []
                    init_atom_y = []
                    init_atom_z = []


                    atom_types_p = []
                    atom_x_p = []
                    atom_y_p = []
                    atom_z_p = []
                    init_atom_types_p = []
                    init_atom_x_p = []
                    init_atom_y_p = []
                    init_atom_z_p = []

                    
                    if loop_num == 0:
                        all_atom = all_atoms[i]
                        mask1 = all_atom.ne(task.dictionary.pad()) 
                        all_atom = all_atom[mask1]  
                        mask2 = all_atom.ne(task.mask_idx)  
                        all_atom = all_atom[mask2]
                        unmasked_atom = all_atom[1:]  
                        unmasked_coord = all_coords[i][mask1][mask2][1:]
                        
                        init_atom = src_tokens[i]
                        mask1_init = init_atom.ne(task.dictionary.pad())  
                        init_atom = init_atom[mask1_init]  
                        mask2_init = init_atom.ne(task.mask_idx)  
                        init_atom = init_atom[mask2_init]
                        unmasked_init_atom = init_atom[1:]  
                        unmasked_init_coord = masked_coord[i][mask1_init][mask2_init][1:]


                    else:
                        unmask_index = all_atoms2[i].ne(task.dictionary.pad()) & all_atoms2[i].ne(model.mask_idx) & all_atoms2[i].ne(model.bos_idx)
                        unmasked_atom = all_atoms[i][unmask_index]  
                        unmasked_coord = all_coords[i][unmask_index]
                    
                    for atom in unmasked_atom:
                        atom_types.append(task.dictionary[atom.item()])

                    for atom in unmasked_init_atom:
                        init_atom_types.append(task.dictionary[atom.item()])

                    for coord in unmasked_coord:
                        atom_x.append(coord[0].item())
                        atom_y.append(coord[1].item())
                        atom_z.append(coord[2].item())

                    for coord in unmasked_init_coord:
                        init_atom_x.append(coord[0].item())
                        init_atom_y.append(coord[1].item())
                        init_atom_z.append(coord[2].item())

                    if loop_num==0:
                        pred_mask = all_atoms[i].eq(task.mask_idx)
                        pred_atom = pred_atoms[i][pred_mask]
                        assert virtual_num ==len(pred_atom), (virtual_num, len(pred_atom), loop_num)
                        pred_coord = pred_coords[i][pred_mask, :]
                        for a in range(len(pred_atom)):
                            if task.dictionary[pred_atom[a].item()] == "[NULL]":
                                assert loop_num==0
                                continue
                            # else:
                            atom_types_p.append(task.dictionary[pred_atom[a].item()])
                            atom_x_p.append(pred_coord[a][0].item())
                            atom_y_p.append(pred_coord[a][1].item())
                            atom_z_p.append(pred_coord[a][2].item())

                        # ## init atoms ## 
                        pred_init_mask = src_tokens[i].eq(task.mask_idx)     
                        pred_init_atom = pred_atoms[i][pred_init_mask]
                        pred_init_coord = pred_coords[i][pred_init_mask, :]
                        for a in range(len(pred_init_atom)):
                            if task.dictionary[pred_init_atom[a].item()] == "[NULL]":
                                assert loop_num==0
                                continue
                            # else:
                            init_atom_types_p.append(task.dictionary[pred_init_atom[a].item()])
                            init_atom_x_p.append(pred_init_coord[a][0].item())
                            init_atom_y_p.append(pred_init_coord[a][1].item())
                            init_atom_z_p.append(pred_init_coord[a][2].item())                   

                    else:
                        pred_mask = all_atoms2[i].eq(task.mask_idx)
                        pred_atom = pred_atoms[i][pred_mask]
                        assert virtual_num ==len(pred_atom), (virtual_num, len(pred_atom), loop_num)
                        pred_coord = pred_coords[i][pred_mask, :]
                        for a in range(len(pred_atom)):
                            if task.dictionary[pred_atom[a].item()] == "[NULL]":
                                assert loop_num==0
                                continue
                            # else:
                            atom_types_p.append(task.dictionary[pred_atom[a].item()])
                            atom_x_p.append(pred_coord[a][0].item())
                            atom_y_p.append(pred_coord[a][1].item())
                            atom_z_p.append(pred_coord[a][2].item())

                    if loop_num==0:
                        lddt_t = 'none'
                    else:
                        lddt_t = str(pred_lddt[i].item()).split('.')[1][:5] 
                    default_add_atom_num = "default_add"
                    default_merge_method = "default_merge"

                    if loop_num == 0:
                        add_atom_num_value = default_add_atom_num
                        merge_method_value = default_merge_method
                    else:
                        add_atom_num_value = str(add_atom_num[i])
                        merge_method_value = str(merge_method[i])

                    outfile = os.path.join(
                        outpath, 
                        str(pdbids[i]), 
                        "{}_final_res{}_{}_{}_{}_{}_loop{}.xyz".format(
                            str(pdbids[i]),
                            str(tt),
                            str(args.seed),
                            add_atom_num_value,
                            merge_method_value,
                            lddt_t,
                            loop_num
                        )
                    )                    

                    tt+=1
                    if not os.path.exists(os.path.join(outpath, str(pdbids[i])) ):
                        try:
                            os.mkdir(os.path.join(outpath, str(pdbids[i])) ) 
                        except:
                            pass
                    title = str(len(atom_types)+len(atom_types_p)) + '\n' + "atom_num:{}|mol_idx:{}".format(len(atom_types)+len(atom_types_p), mol_idx)
                    if loop_num ==0:
                        content = title + "|smiles:" + str(smis[i]) + "|pdbid:" + str(pdbids[i]) + "***"
                    else:
                        if used_atom_num is not None and used_auc is not None:
                            content = title + "|smiles:" + str(smis[i]) + "|pdbid:" + str(pdbids[i]) + '|real_atom:' + str(used_atom_num[i][0]) + '|pred_atom:' +  str(used_atom_num[i][1]) + '|used_atom:' +  str(used_atom_num[i][2]) + '|auc:' + str(used_auc[i]) + '|pred_lddt:' +str(pred_lddt[i].item()) + "\n"
                        else:
                            content = title + "|smiles:" + str(smis[i]) + "|pdbid:" + str(pdbids[i]) + '|real_atom:' + str(0) + '|pred_atom:' +  str(0) + '|used_atom:' +  str(0) + '|auc:' + str(0) + '|pred_lddt:' +str(pred_lddt[i].item()) + "\n"



                    with open(outfile, 'w', encoding='utf-8') as f:
                        f.write(content)
                        for idd in range(len(atom_types)):
                            f.write(str(atom_types[idd]) + " " + str(atom_x[idd]) + " " + str(atom_y[idd]) + " " + str(atom_z[idd]) + "\n")
                        for idd in range(len(atom_types_p)):
                            f.write(str(atom_types_p[idd]) + " " + str(atom_x_p[idd]) + " " + str(atom_y_p[idd]) + " " + str(atom_z_p[idd]) + "\n")
                        f.write("\n")
                    
                    if loop_num == 0:
                        init_outfile = os.path.join(
                            outpath, 
                            str(pdbids[i]), 
                            "{}_init_res{}_{}_{}_{}_{}_loop{}.xyz".format(
                                str(pdbids[i]),
                                str(kk),
                                str(args.seed),
                                add_atom_num_value,
                                merge_method_value,
                                lddt_t,
                                loop_num
                            )
                        )
                        if args.grid_vis:
                            init_grid_outfile = os.path.join(
                            outpath, 
                            str(pdbids[i]), 
                            "{}_init_grid_res{}_{}_{}_{}_{}_loop{}.xyz".format(
                                str(pdbids[i]),
                                str(kk),
                                str(args.seed),
                                add_atom_num_value,
                                merge_method_value,
                                lddt_t,
                                loop_num
                            )
                        )
                            init_grid_noise_outfile = os.path.join(
                            outpath,
                            str(pdbids[i]), 
                            "{}_init_noise_res{}_{}_{}_{}_{}_loop{}.xyz".format(
                                str(pdbids[i]),
                                str(kk),
                                str(args.seed),
                                add_atom_num_value,
                                merge_method_value,
                                lddt_t,
                                loop_num
                            )
                        )

                            element = "H"
                            with open(init_grid_outfile, "w") as file:
                                file.write(f"{len(init_grid[i])}\n")
                                file.write("Grid edges\n")
                                for point in init_grid[i]:
                                    if point[0] == 0 and point[1] == 0 and point[2] == 0:
                                        break
                                    else:
                                        file.write(f"{element} {point[0]} {point[1]} {point[2]}\n")

                            element = "H" 
                            with open(init_grid_noise_outfile, "w") as file:
                                file.write(f"{len(init_grid_noise_atom_coords[i])}\n")
                                file.write("Grid edges\n")
                                for point in init_grid_noise_atom_coords[i]:
                                    if point[0] == 0 and point[1] == 0 and point[2] == 0:
                                        break
                                    else:
                                        file.write(f"{element} {point[0]} {point[1]} {point[2]}\n")                                    


                        kk += 1
                        if not os.path.exists(os.path.join(outpath, str(pdbids[i]))):
                            try:
                                os.makedirs(os.path.join(outpath, str(pdbids[i])))
                            except:
                                pass

                        init_title = str(len(init_atom_types) + len(init_atom_types_p)) + '\n' + "atom_num:{}|mol_idx:{}".format(len(init_atom_types) + len(init_atom_types_p), mol_idx)
                        if loop_num == 0:
                            init_content = init_title + "|smiles:" + str(smis[i]) + "|pdbid:" + str(pdbids[i]) + "***\n"
                        else:
                            init_content = init_title + "|smiles:" + str(smis[i]) + "|pdbid:" + str(pdbids[i]) + '|real_atom:' + str(used_atom_num[i][0] if used_atom_num else 0) + '|pred_atom:' + str(used_atom_num[i][1] if used_atom_num else 0) + '|used_atom:' + str(used_atom_num[i][2] if used_atom_num else 0) + '|auc:' + str(used_auc[i] if used_auc else 0) + '|pred_lddt:' + str(pred_lddt[i].item()) + "\n"
                    

            # print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            
            if _loss is not None and _sample_size is not None and log_output is not None and args.calculate_metric > 0:
                progress.log(log_output, step=j)
                log_outputs.append(log_output)
                # print(log_output)
            else:
                continue

        # save smi_list and embedding_list
        if getattr(args, "embedding_vis", False):
            with open(os.path.join(outpath, f'smi_list_{args.seed}.pkl'), 'wb') as f:
                pickle.dump(np.concatenate(smi_list), f)

            with open(os.path.join(outpath, f'embedding_list_{args.seed}.pkl'), 'wb') as f:
                pickle.dump(np.concatenate(embedding_list), f)    

            with open(os.path.join(outpath, f'embedding_list_loop0_{args.seed}.pkl'), 'wb') as f:
                pickle.dump(np.concatenate(embedding_list_loop0_0), f)

        if data_parallel_world_size > 1:
            # torch.distributed.barrier()
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=45000000,  # args.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            # torch.distributed.barrier()
            log_outputs = list(chain.from_iterable(log_outputs))

        if args.calculate_metric > 0:
            with metrics.aggregate() as agg:
                task.reduce_metrics(log_outputs, loss)
                log_output = agg.get_smoothed_values()

            progress.print(log_output, tag=subset, step=j)



def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    # assert 1==0
    distributed_utils.call_main(
        args, main
    )


if __name__ == "__main__":
    cli_main()
