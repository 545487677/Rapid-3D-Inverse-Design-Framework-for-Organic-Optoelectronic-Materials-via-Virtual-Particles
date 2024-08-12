import logging
import os
import torch
import contextlib
from typing import Optional

import numpy as np
from unicore import checkpoint_utils
from unicore.data import (
    Dictionary,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    TokenizeDataset,
    LMDBDataset,
    EpochShuffleDataset,
    FromNumpyDataset,
    RightPadDataset,
    RightPadDataset2D,
    AppendTokenDataset,
    PrependTokenDataset,
)
from Generator.data import (
    KeyDataset,
    ReorderDataset,
    DistanceDataset,
    RightPadDataset2D2,
    EdgeTypeDataset,
    # TokenizeDataset,
    RightPadDatasetCoord,
    # AtomTypeDataset,
    RemoveHydrogenDataset,
    VirtualPointsSampleDataset,
    PrependAndAppendCentroid
)
from unicore.tasks import UnicoreTask, register_task
import json

logger = logging.getLogger(__name__)


@register_task("unimol_diff_sample")
class UniMolDiffSampleTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.25,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default='normal',
            choices=['trunc_normal', 'uniform', 'normal', 'none'],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--neg-num",
            default=10,
            type=int,
            help="add neg-num blank atoms for each masked atom",
        )
        parser.add_argument(
            "--neg-num-min",
            default=2,
            type=int,
            help="add neg-num blank atoms for each masked atom",
        )
        parser.add_argument(
            "--neg-num-max",
            default=6,
            type=int,
            help="add neg-num blank atoms for each masked atom",
        )
        parser.add_argument(
            "--noise-scale",
            default=0,
            type=float,
            help="add noise for null atom",
        )
        parser.add_argument(
            "--temperature",
            default=100,
            type=float,
            help="add noise for null atom",
        )
        parser.add_argument(
            "--full-gravity",
            default=0,
            type=float,
            help="add full gravity or not",
        )
        parser.add_argument(
            "--post-sample",
            default=0,
            type=int,
            help="post asign coord target or not",
        )
        parser.add_argument(
            "--dict-name",
            default='dict.txt',
            help="dictionary file",
        )
        parser.add_argument(
            "--cos-mask",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--cos-distance",
            type=float,
            default=1,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--cos-bond",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--remove-hydrogen",
            action='store_true',
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action='store_true',
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=510,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--rotation",
            type=float,
            default=0,
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--target-weight",
            type=float,
            default=0.5,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--test-mol",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--l1-coord-loss",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--max-dist",
            type=float,
            default=0.5,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--coord-clamp",
            type=float,
            default=2,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--dist-clamp",
            type=float,
            default=5,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--old-version",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--low-bound",
            type=float,
            default=0.6,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--high-bound",
            type=float,
            default=0.75,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--dynamic-temp",
            type=float,
            default=0.0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--refine-center",
            type=float,
            default=0.0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--pre-set-label",
            type=float,
            default=0.0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--no-teacher-forcing",
            type=float,
            default=0.0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--use-focal-loss",
            type=float,
            default=0.0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--low-neg-bound",
            type=float,
            default=7,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--high-neg-bound",
            type=float,
            default=9,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--not-sample-label",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--sample-greedy",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--calculate-metric",
            default=1,
            type = float,
            help="dictionary file",
        )
        parser.add_argument(
            "--results-name",
            default='res_pbsa',
        )
        parser.add_argument(
            "--use-max-atom",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--use-real-atom",
            type=float,
            default=0,
        )
        parser.add_argument(
            "--use-pred-atom",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--use-add-num",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        )    
        parser.add_argument(
            "--add-atom-num",
            type=float,
            default=0,
            help="mask fragment according to cosine"
        ),
        parser.add_argument(
            "--compress_xy",
            type=float,
            default=0,
            help="compress"
        ),   
        parser.add_argument(
            "--z_rotate",
            type=float,
            default=0,
            help="compress"
        ),   
        parser.add_argument(
            "--compress_optim",
            type=float,
            default=0,
            help="compress"
        ),   
        parser.add_argument(
            "--cubic",
            type=float,
            default=0,
            help="compress"
        ),   
        parser.add_argument(
            "--tune_virtual",
            type=float,
            default=0.0,
            help="compress"
        ),   
        parser.add_argument(
            "--sample_length",
            type=int,
            default=10,
            help="compress"
        ),   
        parser.add_argument(
            "--finetune-encoder-model",
            type=str,
            default=None,
            help="pretrain encoder model path",
        ),
        parser.add_argument(
            "--grid_size",
            type=float,
            default=2,
            help="compress"
        ),  
        parser.add_argument(
            "--grid_offset_size",
            type=float,
            default=0.5,
            help="compress"
        ),  
        parser.add_argument(
            "--pretrain_path",
            type=str,
            default=None,
            help="pretrain encoder model path",
        ),
        parser.add_argument(
            "--grid_vis", action='store_true', help="vis grid"
        ),
        parser.add_argument(
            "--embedding_vis", action='store_true', help="vis grid"
        ),
        parser.add_argument(
            "--optim-frag",
            default=None,
            nargs='+', 
            type=int,
        )

    
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.null_idx = dictionary.add_symbol("[NULL]", is_special=True)
        self.post_sample = args.post_sample # 0
        self.temperature = args.temperature #100.0
        self.full_gravity = args.full_gravity
        if self.args.dict_name == 'new.dict.txt':
            self.use_atom_type = True
        else:
            self.use_atom_type = False

    @classmethod
    def setup_task(cls, args, **kwargs):
        if getattr(args, "finetune_encoder_model", None) is not None or getattr(args, "pretrain_path", None) is not None:
                dictionary = Dictionary.load(os.path.join(args.data, 'pretrain_dict.txt'))
        else:
            dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + '.lmdb')

        # todo: load data from example data
        raw_dataset = LMDBDataset(split_path)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            dataset = ReorderDataset(raw_dataset, coord_seed, 'atoms', 'coordinates', rotation=False)
            dataset = RemoveHydrogenDataset(dataset, 'atoms', 'coordinates', self.args.remove_hydrogen, self.args.remove_polar_hydrogen)
            # dataset = AtomTypeDataset(raw_dataset, dataset, spec=self.use_atom_type)
            token_dataset = KeyDataset(dataset, 'atoms')
            token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
            coord_dataset = KeyDataset(dataset, 'coordinates')
            
            frag_atom_id_dataset = KeyDataset(dataset, 'frag_atom_id')
            # frag_id_dataset = KeyDataset(dataset, 'frag_id')
            frag_coord_dataset = KeyDataset(dataset, 'frag_coord')
            bond_type_dataset = KeyDataset(dataset, 'bond')
            smi_dataset = KeyDataset(dataset, 'smi')
            mol_idx_dataset = KeyDataset(dataset, 'mol_idx')

            

            expand_dataset = VirtualPointsSampleDataset(
                token_dataset,
                coord_dataset,
                frag_atom_id_dataset,
                # frag_id_dataset,
                # frag_coord_dataset,
                bond_type_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                null_idx=self.null_idx,

                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                neg_num=self.args.neg_num,
                temperature = self.temperature,
                args=self.args,
                frag_coord_dataset=frag_coord_dataset,
                dictionary=self.dictionary,
            )
            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            """
            expand_dataset: atoms\atom_tgt\coordinates\coord_tgt\coord_mask            
            """

            add_num_dataset = KeyDataset(expand_dataset, 'add_num')
            pred_num_datatset = KeyDataset(dataset, 'pred_num')

            atom_dataset = KeyDataset(expand_dataset, 'atoms')
            coord_dataset = KeyDataset(expand_dataset, 'coordinates')

            unmask_atom_dataset = KeyDataset(expand_dataset, 'unmask_atom')
            unmask_coord_dataset = KeyDataset(expand_dataset, 'unmask_coord')
            

            virtual_index_dataset1 = KeyDataset(expand_dataset, 'virtual_index')


            unmask_index_dataset = KeyDataset(expand_dataset, 'unmask_index')

            all_atom_dataset = KeyDataset(expand_dataset, 'all_atom')
            all_coord_dataset = KeyDataset(expand_dataset, 'all_coord')
            all_index_dataset = KeyDataset(expand_dataset, 'all_index')
            centroid_dataset = KeyDataset(expand_dataset, 'centroid_dataset')
            centroid_label_dataset = KeyDataset(expand_dataset, 'centroid_label_dataset')

            # src_atom_dataset = PrependAndAppend(atom_dataset, self.dictionary.bos(), self.dictionary.eos())
            src_atom_dataset = PrependTokenDataset(atom_dataset, self.dictionary.bos())
            
            if self.args.refine_center > 0:
                src_coord_dataset = PrependAndAppendCentroid(coord_dataset, centroid_dataset, addeos=False)
            else:
                src_coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0) 

            edge_type = EdgeTypeDataset(src_atom_dataset, len(self.dictionary))
            masked_distance = DistanceDataset(src_coord_dataset)  # dis inp

            if self.args.pre_set_label > 0:
                encoder_null_target = KeyDataset(expand_dataset, 'encoder_null_target')
                coord_null_target = KeyDataset(expand_dataset, 'coord_null_target')
                encoder_null_target_idx = KeyDataset(expand_dataset, 'encoder_null_target_idx')
                merge_idx = KeyDataset(expand_dataset, 'merge_idx')
                merge_weight = KeyDataset(expand_dataset, 'merge_weight')
                encoder_null_target2 = KeyDataset(expand_dataset, 'encoder_null_target2')
                coord_null_target2 = KeyDataset(expand_dataset, 'coord_null_target2')
                pos_hit = KeyDataset(expand_dataset, 'pos_hit')
                virtual_index2 = KeyDataset(expand_dataset, 'virtual_index2')
                src_tokens2 = KeyDataset(expand_dataset, 'src_tokens2')
                src_tokens2 = PrependTokenDataset(src_tokens2, self.dictionary.bos())

                real_pos_hit = KeyDataset(expand_dataset, 'real_pos_hit')
                real_pos_label_hit = KeyDataset(expand_dataset, 'real_pos_label_hit')
                virtual_atom_hit_pre = KeyDataset(expand_dataset, 'virtual_atom_hit_pre')

                coord_null_target_post = KeyDataset(expand_dataset, 'coord_null_target_post')
                coord_null_target_post = PrependTokenDataset(coord_null_target_post, 0.0)


                target = {
                        'unmask_atom': RightPadDataset(unmask_atom_dataset, pad_idx=self.dictionary.pad()),
                        'unmask_coord': RightPadDatasetCoord(unmask_coord_dataset, pad_idx=0),
                        'virtual_index': RightPadDataset(virtual_index_dataset1, pad_idx=self.dictionary.pad()),
                        'unmask_index': RightPadDataset(unmask_index_dataset, pad_idx=self.dictionary.pad()),      
                        'all_atom': RightPadDataset(all_atom_dataset, pad_idx=self.dictionary.pad()),
                        'all_coord': RightPadDatasetCoord(all_coord_dataset, pad_idx=0),
                        'all_index': RightPadDataset(all_index_dataset, pad_idx=self.dictionary.pad()),
                        'centroid_label_dataset': centroid_label_dataset,
                        'encoder_null_target': RightPadDataset(encoder_null_target, pad_idx=self.dictionary.pad()),
                        'coord_null_target': RightPadDatasetCoord(coord_null_target, pad_idx=0),
                        'encoder_null_target_idx': RightPadDataset(encoder_null_target_idx, pad_idx=self.dictionary.pad()),
                        'coord_index': RightPadDataset2D2(merge_idx, pad_idx=0),
                        'index_weight': RightPadDataset2D2(merge_weight, pad_idx=0),
                        'encoder_null_target2': RightPadDataset(encoder_null_target2, pad_idx=self.dictionary.pad()),
                        'coord_null_target2': RightPadDatasetCoord(coord_null_target2, pad_idx=0),
                        'pos_hit': pos_hit,
                        'virtual_index2': RightPadDataset(virtual_index2, pad_idx=self.dictionary.pad()),
                        'src_tokens2': RightPadDataset(src_tokens2, pad_idx=self.dictionary.pad()),
                        'real_pos_hit': real_pos_hit,
                        'real_pos_label_hit': real_pos_label_hit,
                        'virtual_atom_hit_pre': virtual_atom_hit_pre,
                        
                   }
                if self.args.grid_vis:
                    grid_coords_dataset = KeyDataset(expand_dataset, 'grid_coords')
                    grid_noise_atom_coords_dataset = KeyDataset(expand_dataset, 'grid_noise_atom_coords')

                    target['grid_coords'] = RightPadDatasetCoord(grid_coords_dataset, pad_idx=0)  
                    target['grid_noise_atom_coords'] = RightPadDatasetCoord(grid_noise_atom_coords_dataset, pad_idx=0) 


            else:
                target = {
                       'unmask_atom':
                           RightPadDataset(unmask_atom_dataset, pad_idx=self.dictionary.pad()),
                       'unmask_coord':
                           RightPadDatasetCoord(unmask_coord_dataset, pad_idx=0),
                       'virtual_index':
                           RightPadDataset(virtual_index_dataset1, pad_idx=self.dictionary.pad()),
                       'unmask_index':
                           RightPadDataset(unmask_index_dataset, pad_idx=self.dictionary.pad()),      
                        'all_atom':
                           RightPadDataset(all_atom_dataset, pad_idx=self.dictionary.pad()),
                        'all_coord':
                           RightPadDatasetCoord(all_coord_dataset, pad_idx=0),
                        'all_index':
                           RightPadDataset(all_index_dataset, pad_idx=self.dictionary.pad()),
                        'centroid_label_dataset': centroid_label_dataset,
                   }


            return {
                       "src_tokens": RightPadDataset(
                           src_atom_dataset,
                           pad_idx=self.dictionary.pad(),
                       ),
                       'masked_distance': RightPadDataset2D(
                           masked_distance,
                           pad_idx=0,
                       ),
                       'masked_coord': RightPadDatasetCoord(
                           src_coord_dataset,
                           pad_idx=0,
                       ),
                       'edge_type': RightPadDataset2D(
                           edge_type,
                           pad_idx=0,
                       ),
                       'smi': smi_dataset,
                       'mol_idx': mol_idx_dataset,
                       'coord_null_target_post':RightPadDatasetCoord(
                            coord_null_target_post, 
                            pad_idx=0
                        ),
                        'pred_atom_num_mean2': pred_num_datatset,
                        'add_atom_num_all': add_num_dataset,
                        
                   }, target

        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {'net_input': net_input, 'target': target}
        if self.args.contrastive_loss > 0:
            cl_net_input, cl_target = one_dataset(raw_dataset, self.args.seed + 42, self.args.seed)
            dataset['cl_net_input'] = cl_net_input
            dataset['cl_target'] = cl_target
        dataset = NestedDictionaryDataset(
            dataset
        )
        if split in ['train', 'train.small']:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def inference_step(self, sample, model, loss, nsample):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output, all_output = loss(model, sample, inference=True, nsample=nsample)
        return loss, sample_size, logging_output, all_output

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.finetune_encoder_model is not None:
                print("load pretrain model weight from...", args.finetune_encoder_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_encoder_model,
                )
                new_state_dict1 = state["model"].copy()
                new_state_dict1 = {k.replace('encoder.', 'encoder1.'): v for k, v in new_state_dict1.items()}
                model.load_state_dict(new_state_dict1, strict=False)
                new_state_dict2 = state["model"].copy()
                new_state_dict2 = {k.replace('encoder.', 'encoder2.'): v for k, v in new_state_dict2.items()}                
                model.load_state_dict(new_state_dict2, strict=False)                
                return model
        # elif args.pretrain_path is not None:
        #     print("load pretrain model weight from...", args.pretrain_path)
        #     state = checkpoint_utils.load_checkpoint_to_cpu(
        #         args.pretrain_path,
        #     )
        #     model.load_state_dict(state["model"], strict=False)
        #     return model
        else:
            return model