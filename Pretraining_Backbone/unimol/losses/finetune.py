# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import r2_score
import numpy as np

ATTR2IDX = {
    'eps0814': {'target': 0},
    'thermo0814': {'target': 0},
    'visco0814': {'target': 0},
    'visco0830': {'target': 0},
    'visco0911': {'target': 0},
    'visco0915': {'target': 0},
    'visco0916': {'target': 0},

}
ATTR_REGESTRY = {
    # mean, std, type

    'eps0814': [np.array([5.22214]),
                np.array([2.7061221082254847]),
                'standardization'],
    'thermo0814': [np.array([0.13576349206349209]),
                   np.array([0.03174528757744114]),
                   'standardization'],
    'visco0814': [np.array([1.3427064676616915]),
                  np.array([1.44283092]),
                  'standardization'],
    'visco0830': [np.array([5.475559523809532]),
                  np.array([45.86006025204235]),
                  'standardization'],
    'visco0911': [np.array([4.687795939086294]),
                  np.array([33.6399336175123]),
                  'standardization'],
    'visco0915': [np.array([1.4116399572649574]),
                  np.array([1.6776511787964796]),
                  'standardization'],
    'visco0916': [np.array([1.6473238993710693]),
                  np.array([2.409164549978287]),
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
            raise ValueError(
                'normal_type should be log1p_standardization or standardization')

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
            raise ValueError(
                'normal_type should be log1p_standardization or standardization')


@register_loss("finetune_mae")
class FinetuneMAELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.target_list = self.args.target_name.split(',')
        self.attr2idx = ATTR2IDX[self.args.task_name]

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
        )
        net_output = net_output[0]
        assert isinstance(net_output, dict), "net_output should be a dict"
        predicts = net_output[self.args.classification_head_name]
        sample_size = predicts.size(0)
        _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
        _mean = torch.tensor(_mean).to(predicts.device).float()
        _std = torch.tensor(_std).to(predicts.device).float()
        normalizer = Normalization(_mean, _std, _normal_type)
        target_loss, dist_loss, coord_loss = self.compute_loss(
            model, net_output, sample, normalizer)
        if target_loss is None:
            target_loss = torch.Tensor([0]).to(predicts.device)
        if dist_loss is None:
            dist_loss = torch.Tensor([0]).to(predicts.device)
        if coord_loss is None:
            coord_loss = torch.Tensor([0]).to(predicts.device)
        loss = target_loss + dist_loss + coord_loss
        if not self.training:
            logging_output = {
                "smi": sample["smi"],
                "loss": loss.data,
                "target_loss": target_loss.data,
                "dist_loss": dist_loss.data,
                "coord_loss": coord_loss.data,
                "predict": normalizer.inverse_transform(predicts.view(-1, self.args.num_classes).data),
                "sample_size": sample_size,
                "bsz": sample_size,
                "target_list": self.target_list,
                "task_name": self.args.task_name,
            }
            if 'target' in sample and 'finetune_target' in sample['target']:
                logging_output['target'] = sample["target"]["finetune_target"][0].view(
                    -1, self.args.num_classes).data
            # if self.args.aux_coord_loss and 'coord_s0' in net_output and 'coord_s1' in net_output:
            #     logging_output['predict_coord_s0'] = net_output['coord_s0'].float().squeeze(-1).data
            #     logging_output['predict_coord_s1'] = net_output['coord_s1'].float().squeeze(-1).data
            #     logging_output['target_coord_s0'] = sample['target']['coord_s0_target'].float().squeeze(-1).data
            #     logging_output['target_coord_s1'] = sample['target']['coord_s1_target'].float().squeeze(-1).data
            #     logging_output['coord_center'] = sample['coord_center']
            if self.args.aux_dist_loss and 'distance_s0' in net_output and 'distance_s1' in net_output:

                logging_output['target_dist_s0'] = sample['target']['distance_s0_target'].float(
                ).squeeze(-1).data
                logging_output['target_dist_s1'] = sample['target']['distance_s1_target'].float(
                ).squeeze(-1).data

                if self.args.opt_coord:
                    logging_output['predict_dist_s0'] = net_output['distance_s0'][0].float(
                    ).squeeze(-1).data
                    logging_output['predict_dist_s1'] = net_output['distance_s1'][0].float(
                    ).squeeze(-1).data
                if not self.args.opt_coord:
                    logging_output['predict_dist_s0'] = net_output['distance_s0'].float(
                    ).squeeze(-1).data
                    logging_output['predict_dist_s1'] = net_output['distance_s1'].float(
                    ).squeeze(-1).data
                if self.args.opt_coord or self.args.aux_coord_loss:
                    logging_output['predict_coord_s0'] = net_output['distance_s0'][1].float(
                    ).squeeze(-1).data
                    logging_output['predict_coord_s1'] = net_output['distance_s1'][1].float(
                    ).squeeze(-1).data
                    logging_output['target_coord_s0'] = sample['target']['coord_s0_target'].float(
                    ).squeeze(-1).data
                    logging_output['target_coord_s1'] = sample['target']['coord_s1_target'].float(
                    ).squeeze(-1).data
                    logging_output['coord_center'] = sample['coord_center']

        else:
            logging_output = {
                "loss": loss.data,
                "target_loss": target_loss.data,
                "dist_loss": dist_loss.data,
                "coord_loss": coord_loss.data,
                "sample_size": sample_size,
                "bsz": sample_size,
            }
        return loss, 1, logging_output

    def compute_loss(self, model, net_output, sample, normalizer):
        loss, dist_loss, coord_loss = None, None, None

        # target loss
        if 'target' in sample and 'finetune_target' in sample['target']:
            predicts = net_output[self.args.classification_head_name].view(
                -1, self.args.num_classes).float()
            # print("pred {}".format(predicts))
            targets = sample['target']['finetune_target'][0].view(
                -1, self.args.num_classes).float()
            normalize_targets = normalizer.transform(targets).float()
            # print('target {}'.format(targets))
            # print('normalize target {}'.format(normalize_targets))
            loss = F.mse_loss(
                predicts,
                normalize_targets,
                reduction="mean",
            )

        # aux dist loss
        if self.args.aux_dist_loss and 'distance_s0' in net_output and 'distance_s1' in net_output:
            if self.args.opt_coord:
                predict_dist_s0 = net_output['distance_s0'][0].float(
                ).squeeze(-1)
                predict_dist_s1 = net_output['distance_s1'][0].float(
                ).squeeze(-1)
            else:
                predict_dist_s0 = net_output['distance_s0'].float().squeeze(-1)
                predict_dist_s1 = net_output['distance_s1'].float().squeeze(-1)
            targets_dist_s0 = sample['target']['distance_s0_target'].float(
            ).squeeze(-1)
            targets_dist_s1 = sample['target']['distance_s1_target'].float(
            ).squeeze(-1)
            mask_s0 = targets_dist_s0 != 0
            mask_s1 = targets_dist_s1 != 0
            dist_loss_s0 = F.l1_loss(
                predict_dist_s0[mask_s0],
                targets_dist_s0[mask_s0],
                reduction="mean",
            )
            dist_loss_s1 = F.l1_loss(
                predict_dist_s1[mask_s1],
                targets_dist_s1[mask_s1],
                reduction="mean",
            )
            dist_loss = dist_loss_s0 + dist_loss_s1

        # aux coord loss
        # if self.args.aux_coord_loss and 'coord_s0' in net_output and 'coord_s1' in net_output:
        if self.args.aux_dist_loss and self.args.opt_coord:
            predict_coord_s0 = net_output['distance_s0'][1].float().squeeze(-1)
            predict_coord_s1 = net_output['distance_s0'][1].float().squeeze(-1)
            targets_coord_s0 = sample['target']['coord_s0_target'].float(
            ).squeeze(-1)
            targets_coord_s1 = sample['target']['coord_s1_target'].float(
            ).squeeze(-1)
            mask_s0 = targets_coord_s0 != 0
            mask_s1 = targets_coord_s1 != 0
            coord_loss_s0 = F.l1_loss(
                predict_coord_s0[mask_s0],
                targets_coord_s0[mask_s0],
                reduction="mean",
            )
            coord_loss_s1 = F.l1_loss(
                predict_coord_s1[mask_s1],
                targets_coord_s1[mask_s1],
                reduction="mean",
            )
            coord_loss = coord_loss_s0 + coord_loss_s1

        return loss, dist_loss, coord_loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        target_loss_sum = sum(log.get("target_loss", 0)
                              for log in logging_outputs)
        dist_loss_sum = sum(log.get("dist_loss", 0) for log in logging_outputs)
        coord_loss_sum = sum(log.get("coord_loss", 0)
                             for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=4
        )
        metrics.log_scalar(
            "target_loss", target_loss_sum / sample_size, sample_size, round=4
        )
        if dist_loss_sum > 0:
            metrics.log_scalar(
                "dist_loss", dist_loss_sum / sample_size, sample_size, round=4
            )
        if coord_loss_sum > 0:
            metrics.log_scalar(
                "coord_loss", coord_loss_sum / sample_size, sample_size, round=4
            )
        if "valid" in split or "test" in split:
            if "target" in logging_outputs[0]:
                predicts = torch.cat([log.get("predict")
                                     for log in logging_outputs], dim=0)
                targets = torch.cat([log.get("target")
                                    for log in logging_outputs], dim=0)
                sz = predicts.size(0)
                predicts = predicts.view(sz, ).cpu().numpy()
                targets = targets.view(sz, ).cpu().numpy()
                r2 = r2_score(targets, predicts)
                metrics.log_scalar(f"{split}_r2", r2, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
