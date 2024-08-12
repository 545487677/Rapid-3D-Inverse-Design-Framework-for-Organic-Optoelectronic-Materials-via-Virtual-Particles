# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import r2_score

ATTR_REGESTRY = {
    'energy': [np.array([-10.150410724524885, -8.954623785706517, -4759.428985003342]), 
                np.array([0.4663760450930604, .5649460324806638, 995.617059574614]), 
                'standardization'],
}

class Normalization(object):
    def __init__(self, mean=None, std=None, normal_type=None):
        self.mean = mean
        self.std = std
        self.normal_type = normal_type
    
    def transform(self, x):
        if self.normal_type == 'log1p_standardization':
            return (torch.log1p(x) - self.mean) / self.std
        elif self.normal_type == 'standardization':
            return (x - self.mean) / self.std
        elif self.normal_type == 'centering':
            return x - self.mean
        elif self.normal_type == 'none':
            return x
        else:
            raise ValueError('normal_type should be log1p_standardization or standardization')
    
    def inverse_transform(self, x):
        if self.normal_type == 'log1p_standardization':
            return torch.expm1(x * self.std + self.mean)
        elif self.normal_type == 'standardization':
            return x * self.std + self.mean
        elif self.normal_type == 'centering':
            return x + self.mean
        elif self.normal_type == 'none':
            return x
        else:
            raise ValueError('normal_type should be log1p_standardization or standardization')
        
@register_loss("unimol")
class UniMolLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()
        logits_encoder, encoder_distance, encoder_coord, encoder_charge, energy_pred, x_norm, delta_encoder_pair_rep_norm = model(**sample[input_key], encoder_masked_tokens=masked_tokens)
        target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = target[masked_tokens]
        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = masked_token_loss * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1) * sample[target_key]["tokens_target"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
        }

        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_distance = encoder_distance[dist_masked_tokens, :]
            masked_distance_target = sample[target_key]["distance_target"][dist_masked_tokens]
            non_pad_pos = masked_distance_target > 0
            non_pad_pos &= masked_distance_target < self.args.dist_threshold
            masked_dist_loss = F.smooth_l1_loss(
                masked_distance[non_pad_pos].view(-1).float(),
                masked_distance_target[non_pad_pos].view(-1),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if encoder_coord is not None:
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if encoder_charge is not None:
            charge_target = sample[target_key]["charge_target"]
            masked_charge_loss = F.smooth_l1_loss(
                encoder_charge[masked_tokens].view(-1, 1).float(),
                charge_target[masked_tokens].view(-1, 1),
                reduction="mean",
                beta=0.25,
            )  
            loss = loss + masked_charge_loss * self.args.masked_charge_loss
            logging_output["masked_charge_loss"] = masked_charge_loss.data

        if energy_pred is not None:
            energy_target = sample[target_key]["energy_target"]
            _mean, _std, _normal_type = ATTR_REGESTRY["energy"]
            _mean = torch.tensor(_mean).to(energy_pred.device).float()
            _std = torch.tensor(_std).to(energy_pred.device).float()
            normalizer = Normalization(_mean, _std, _normal_type)
            normalize_energy_target = normalizer.transform(energy_target)
            energy_loss = F.smooth_l1_loss(
                energy_pred.view(-1, 3).float(),
                normalize_energy_target.view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + energy_loss * self.args.energy_loss
            logging_output["energy_loss"] = energy_loss.data
            if not self.training:
                logging_output["energy_pred"] = normalizer.inverse_transform(energy_pred.data)
                logging_output["energy_target"] = energy_target.data

        if x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if delta_encoder_pair_rep_norm is not None:
            loss = loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            logging_output["delta_pair_repr_norm_loss"] = delta_encoder_pair_rep_norm.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "seq_len", seq_len / bsz, 1, round=3
        )

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar("masked_token_loss", masked_loss / sample_size, sample_size, round=3)

        masked_acc = sum(log.get("masked_token_hit", 0) for log in logging_outputs) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        masked_coord_loss = sum(log.get("masked_coord_loss", 0) for log in logging_outputs)
        if masked_coord_loss > 0:
            metrics.log_scalar("masked_coord_loss", masked_coord_loss / sample_size, sample_size, round=3)

        masked_dist_loss = sum(log.get("masked_dist_loss", 0) for log in logging_outputs)
        if masked_dist_loss > 0:
            metrics.log_scalar("masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3)
        
        masked_charge_loss = sum(log.get("masked_charge_loss", 0) for log in logging_outputs)
        if masked_charge_loss > 0:
            metrics.log_scalar("masked_charge_loss", masked_charge_loss / sample_size, sample_size, round=3)
        
        energy_loss = sum(log.get("energy_loss", 0) for log in logging_outputs)
        if energy_loss > 0:
            metrics.log_scalar("energy_loss", energy_loss / sample_size, sample_size, round=3)
            if split in ['valid', 'test']:
                energy_pred = torch.cat([log.get("energy_pred", 0) for log in logging_outputs], dim=0).cpu().numpy()
                energy_target = torch.cat([log.get("energy_target", 0) for log in logging_outputs], dim=0).cpu().numpy()
                metrics.log_scalar("homo_r2", r2_score(energy_target[:,0], energy_pred[:,0]), 1, round=3)
                metrics.log_scalar("lumo_r2", r2_score(energy_target[:,1], energy_pred[:,1]), 1, round=3)
                metrics.log_scalar("e_r2", r2_score(energy_target[:,2], energy_pred[:,2]), 1, round=3)

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar("x_norm_loss", x_norm_loss / sample_size, sample_size, round=3)

        delta_pair_repr_norm_loss = sum(log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs)
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar("delta_pair_repr_norm_loss", delta_pair_repr_norm_loss / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
