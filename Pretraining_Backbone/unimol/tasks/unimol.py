# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
)
from unimol.data import (
    KeyDataset,
    RawListDataset,
    MaskPointsDataset,
    DistanceDataset,
    EdgeTypeDataset,
    ConformerSampleDataset,
    PrependAndAppend2DDataset,
    PrependAndAppendPairChargeDataset,
    RightPadDatasetCoord,
    RightPadDataset2D,
    CroppingDataset,
    NormalizeDataset,
    LMDBDataset,
    PairChargeDataset,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


@register_task("unimol")
class UniMolTask(UnicoreTask):
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
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
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
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=1024,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=8.0,
            help="distance threshold for distance loss",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        dataset = ConformerSampleDataset(dataset, self.args.seed, 'atoms',
                                         'coordinates')
#        energy_target = KeyDataset(dataset, "energy")
#        smi = KeyDataset(dataset, "smi")
#        charge_dataset = KeyDataset(dataset, "atoms_charge")
        dataset = CroppingDataset(dataset, self.args.seed, 'atoms',
                                  'coordinates', self.args.max_atoms)
        dataset = NormalizeDataset(dataset, 'coordinates')
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset,
                                        self.dictionary,
                                        max_seq_len=self.args.max_seq_len)
        coord_dataset = KeyDataset(dataset, "coordinates")
        expand_dataset = MaskPointsDataset(
            token_dataset,
            coord_dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
        encoder_target_dataset = KeyDataset(expand_dataset, "targets")
        encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")
#        encoder_charge_dataset = KeyDataset(expand_dataset, "atoms_charge")

        src_dataset = PrependAndAppend(encoder_token_dataset,
                                       self.dictionary.bos(),
                                       self.dictionary.eos())
        tgt_dataset = PrependAndAppend(encoder_target_dataset,
                                       self.dictionary.pad(),
                                       self.dictionary.pad())

        encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)
        encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0,
                                                 0.0)
        encoder_distance_dataset = PrependAndAppend2DDataset(
            encoder_distance_dataset, 0.0)
#        encoder_pair_charge_dataset = PairChargeDataset(encoder_charge_dataset)
#        encoder_pair_charge_dataset = PrependAndAppendPairChargeDataset(
#            encoder_pair_charge_dataset, 0.0)

        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
#        charge_dataset = RawListDataset(charge_dataset)
#        charge_dataset = PrependAndAppend(charge_dataset, 0.0, 0.0)
        net_input = {
            "src_tokens":
            RightPadDataset(
                src_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "src_coord":
            RightPadDatasetCoord(
                encoder_coord_dataset,
                pad_idx=0,
            ),
            "src_distance":
            RightPadDataset2D(
                encoder_distance_dataset,
                pad_idx=0,
            ),
            "src_edge_type":
            RightPadDataset2D(
                edge_type,
                pad_idx=0,
            ),

            #            "smi":
            #            smi,
        }
        target = {
            "tokens_target":
            RightPadDataset(tgt_dataset, pad_idx=self.dictionary.pad()),
            "distance_target":
            RightPadDataset2D(distance_dataset, pad_idx=0),
            "coord_target":
            RightPadDatasetCoord(coord_dataset, pad_idx=0),
            #            "charge_target":
            #            RightPadDataset(charge_dataset, pad_idx=0),
            #            "energy_target":
            #            RawListDataset(energy_target),
        }
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset),
                                          self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model
