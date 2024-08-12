# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import pandas as pd
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    EpochShuffleDataset,
    TokenizeDataset,
)
from unimol.data import (
    KeyDataset,
    EdgeTypeDataset,
    ConformerSampleDataset,
    RightPadDatasetCoord,
    CroppingDataset,
    NormalizeDataset,
    NormalizeAndAlignDataset,
    LMDBDataset,
    RightPadDataset,
    PairChargeDataset,
    PrependAndAppendPairChargeDataset,
    DistanceDataset,
    PrependAndAppend2DDataset,
    RawListDataset,
    RightPadDataset2D,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


@register_task("unimol_finetune")
class UniMolFinetuneTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path"
        )
        parser.add_argument(
            "--task-name",
            type=str,
            default='',
            help="downstream task name"
        )
        parser.add_argument(
            "--fold",
            type=int,
            default=0,
            help="downstream task fold"
        )
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune classification task name"
        )
        parser.add_argument(
            "--distance-head-name",
            default="distance_s0,distance_s1",
            # split by ',' if multiple distance heads
            help="finetune distance head task name"
        )
        parser.add_argument(
            "--coord-head-name",
            default="coord_s0,coord_s1",
            help="finetune coord head task name"   # split by ',' if multiple coord heads
        )
        parser.add_argument(
            "--num-classes",
            default=5,
            type=int,
            help="finetune downstream task classes numbers"
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--target-name",
            help="choose to use target names",
        )
        parser.add_argument(
            "--aux-dist-loss",
            action="store_true",
            help="use auxiliary distance loss",
        )
        parser.add_argument(
            "--opt-coord",
            action="store_true",
            help="distance loss with optimize coordinates",
        )
        parser.add_argument(
            "--aux-coord-loss",
            action="store_true",
            help="use auxiliary coord loss",
        )
        parser.add_argument(
            "--inference",
            action="store_true",
            help="inference mode",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(
            self.args.data, self.args.task_name, f'fold{self.args.fold}', split + ".lmdb")
        dataset = LMDBDataset(split_path)
        dataset = ConformerSampleDataset(
            dataset, self.args.seed, 'atoms', 'coordinates')
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms)
        # dataset = NormalizeDataset(dataset, "coordinates")
        if not self.args.inference and (self.args.aux_dist_loss or self.args.aux_coord_loss):
            dataset = NormalizeAndAlignDataset(
                dataset, "coordinates", "coordinates_s0", "coordinates_s1")
            coord_center_dataset = KeyDataset(dataset, 'coord_center')
        else:
            dataset = NormalizeDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        smi_dataset = KeyDataset(dataset, "smi")
#        charge_dataset = KeyDataset(dataset, "atoms_charge")
#        charge_dataset = RawListDataset(charge_dataset)
        if not self.args.inference:
            tgt_dataset = KeyDataset(dataset, "target")
        token_dataset = KeyDataset(dataset, "atoms")
        coord_dataset = KeyDataset(dataset, "coordinates")
        if not self.args.inference and (self.args.aux_dist_loss or self.args.aux_coord_loss):
            coord_s0_dataset = KeyDataset(dataset, "coordinates_s0")
            coord_s1_dataset = KeyDataset(dataset, "coordinates_s1")
            coord_s0_dataset = RawListDataset(coord_s0_dataset)
            coord_s1_dataset = RawListDataset(coord_s1_dataset)

        token_dataset = TokenizeDataset(
            token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        src_dataset = PrependAndAppend(
            token_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        distance_dataset = DistanceDataset(coord_dataset)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)

#        pair_charge_dataset = PairChargeDataset(charge_dataset)
#        pair_charge_dataset = PrependAndAppendPairChargeDataset(pair_charge_dataset, 0.0)
        if not self.args.inference and (self.args.aux_dist_loss or self.args.aux_coord_loss):
            distance_s0_dataset = DistanceDataset(coord_s0_dataset)
            distance_s0_dataset = PrependAndAppend2DDataset(
                distance_s0_dataset, 0.0)
            coord_s0_dataset = PrependAndAppend(coord_s0_dataset, 0.0, 0.0)
            distance_s1_dataset = DistanceDataset(coord_s1_dataset)
            distance_s1_dataset = PrependAndAppend2DDataset(
                distance_s1_dataset, 0.0)
            coord_s1_dataset = PrependAndAppend(coord_s1_dataset, 0.0, 0.0)

        target_dd = {}
        if not self.args.inference:
            target_dd.update({"finetune_target": tgt_dataset})
            if self.args.aux_dist_loss:
                target_dd.update({"distance_s0_target": RightPadDataset2D(
                    distance_s0_dataset,
                    pad_idx=0)
                })
                target_dd.update({"distance_s1_target": RightPadDataset2D(
                    distance_s1_dataset,
                    pad_idx=0)
                })
            if self.args.aux_coord_loss or self.args.opt_coord:
                target_dd.update({"coord_s0_target": RightPadDatasetCoord(
                    coord_s0_dataset,
                    pad_idx=0)
                })
                target_dd.update({"coord_s1_target": RightPadDatasetCoord(
                    coord_s1_dataset,
                    pad_idx=0)
                })

        net_input = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "src_coord": RightPadDatasetCoord(
                coord_dataset,
                pad_idx=0,
            ),
            "src_distance": RightPadDataset2D(
                distance_dataset,
                pad_idx=0,
            ),
            "src_edge_type": RightPadDataset2D(
                edge_type,
                pad_idx=0,
            ),
            # "src_pair_charge": RightPadDataset2D(
            #     pair_charge_dataset,
            #     pad_idx=0,
            # ),
            "smi": smi_dataset,
        }
        dd = {
            "net_input": net_input,
            "target": target_dd,
            "smi": KeyDataset(dataset, "smi"),

        }
        if not self.args.inference and (self.args.aux_dist_loss or self.args.aux_coord_loss):
            dd.update({"coord_center": coord_center_dataset})

        nest_dataset = NestedDictionaryDataset(dd)

        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        if self.args.aux_dist_loss:
            distance_name_list = self.args.distance_head_name.split(',')
            if self.args.opt_coord:
                for distance_name in distance_name_list:
                    model.register_distance_head(
                        distance_name,
                        1,
                        True,
                    )
            else:
                for distance_name in distance_name_list:
                    model.register_distance_head(
                        distance_name,
                        1,
                        False,
                    )
        if self.args.aux_coord_loss:
            coord_name_list = self.args.coord_head_name.split(',')
            for coord_name in coord_name_list:
                model.register_coord_head(
                    coord_name,
                )
        return model
