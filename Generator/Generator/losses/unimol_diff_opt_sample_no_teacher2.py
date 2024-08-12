import math
import torch
import torch.nn.functional as F
from unicore import metrics, utils
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from sklearn import metrics as sk_metrics
from typing import List, Callable, Any, Dict


@register_loss("unimol_diff_opt_sample_e2e2")
class UniMolDiffOptSamplee2e2Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.mask_idx = task.mask_idx
        self.null_idx = task.null_idx
        self.full_gravity = task.full_gravity
        if self.args.contrastive_loss > 0:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.cl_label = None
        self.temperature = task.temperature
        self.relu = torch.nn.ReLU()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.gamma = 2

    def forward(self, model, sample, reduce=True, inference=False, nsample=0, add_atom=0):
        def inner_forward(input_key='net_input', target_key='target', inference=False):


            repeated_num = self.args.method_num

            unmask_atom = sample[target_key]['unmask_atom'] 
            unmask_coord = sample[target_key]['unmask_coord'] 
            unmask_index = sample[target_key]['unmask_index']
            masked_tokens = unmask_index.eq(0) 
            masked_tokens_pred = unmask_index.ne(0)



            src_tokens2 = sample[target_key]['src_tokens2'] 
            coord_index = sample[target_key]['coord_index']
            index_weight = sample[target_key]['index_weight'] 
            all_atom = sample[target_key]['all_atom']


            all_loss = model(**sample[input_key], masked_tokens=None, mask_idx=self.mask_idx, inference=inference, nsample=nsample, src_tokens2=src_tokens2, coord_index=coord_index, index_weight=index_weight, all_atom=all_atom, no_teacher_forcing=self.args.no_teacher_forcing>0, add_atom=self.args.add_atom_num )
            whole_loss = 0
            logging_output = {}
            ouput_for_inference = []


            for i, _loss in enumerate(all_loss): 
                logits_encoder, encoder_coord, encoder_null_coord_pre, x_norm, delta_pair_repr_norm, encoder_pair_rep, pre_coord_init, pred_dist_loss, src_tokens, masked_coord, pred_merge, pred_atom_num, encoder_pocket_pair_rep, src_tokens_merge, virtual_index2, merge_label, used_atom_num, used_auc, null_pred, pred_null_dis, add_atom_num, merge_method, src_tokens, init_coord, encoder_rep  = _loss
                if inference:
                    if i==0:
                        ouput_for_inference.append([ logits_encoder, encoder_coord, src_tokens, masked_coord,  virtual_index2, None, None, None, None , add_atom_num, merge_method, src_tokens, init_coord, encoder_rep])
                    else:
                        ouput_for_inference.append([ logits_encoder, encoder_coord, src_tokens_merge, masked_coord,  virtual_index2, src_tokens, used_atom_num, used_auc, add_atom_num, merge_method, src_tokens, init_coord, encoder_rep])
                        
                
                output_atom = logits_encoder.clone()
                output_coord = encoder_coord.clone()

                if i==0:
                    virtual_index = sample[target_key]['virtual_index']
                    virtual_size = virtual_index.shape[1]

                    encoder_null_target = sample[target_key]['encoder_null_target']
                    coord_null_target = sample[target_key]['coord_null_target'] 
                    encoder_null_target_idx = sample[target_key]['encoder_null_target_idx']
                else:
                    virtual_index = virtual_index2
                    virtual_size = virtual_index.shape[1]

                    encoder_null_target = None
                    coord_null_target = None

            
                assert src_tokens.size(1) >= virtual_size, (src_tokens.shape, virtual_index.shape, virtual_size,i)
                
                if logits_encoder.size(1) == virtual_size:
                    virtual_index = virtual_index[:,:-1]
                    if encoder_null_target is not None:
                        assert (encoder_null_target[:,-1].eq(self.padding_idx)).all()
                        encoder_null_target = encoder_null_target[:,:-1]
                        coord_null_target = coord_null_target[:,:-1]
                        encoder_null_target_idx = encoder_null_target_idx[:,:-1]
                    

                null_tokens = virtual_index.eq(0)
                null_tokens_pred = virtual_index.ne(0)

                logits_encoder_null = logits_encoder[:,1:virtual_size+1, :] ##
                encoder_null_coord = encoder_coord[:,1:virtual_size+1,:] ##

                if null_pred is not None:
                    null_pred = null_pred[:,1:virtual_size+1,:] 

                if pred_null_dis is not None:
                    pred_null_dis = pred_null_dis[:,1:virtual_size+1,:] 

                null_tokens = virtual_index.eq(0)
                null_tokens_pred = virtual_index.ne(0) 

                
                encoder_null_coord_pre = encoder_null_coord_pre[:, 1:virtual_size+1, :]
                pre_coord = pre_coord_init[:, 1:virtual_size+1, :]
                if pred_dist_loss is not None:
                    pred_dist_loss = pred_dist_loss[:, 1:virtual_size+1, :]
                assert encoder_null_coord_pre.size(1) == pre_coord.size(1)

                
                
                if unmask_coord.size(0)!=encoder_null_coord.size(0):
                    unmask_coord = unmask_coord.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,unmask_coord.size(1),unmask_coord.size(2))
                    unmask_atom = unmask_atom.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,unmask_atom.size(1))
                    unmask_index = unmask_index.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,unmask_index.size(1)) 
                    masked_tokens = unmask_index.eq(0)
                    masked_tokens_pred = unmask_index.ne(0)
                
                distance_null_to_mask_test = torch.norm((encoder_null_coord.unsqueeze(2) - unmask_coord.unsqueeze(1)), dim=-1)
                distance_null_to_mask_test.masked_fill_(
                    masked_tokens.unsqueeze(1).to(torch.bool),
                    100000
                )
                distance_null_to_mask_test = distance_null_to_mask_test.permute(0, 2, 1)
                distance_null_to_mask_test.masked_fill_(
                    null_tokens.unsqueeze(1).to(torch.bool),
                    100000,
                )
                distance_null_to_mask_test = distance_null_to_mask_test.permute(0, 2, 1)
                distance_null_to_mask_index_test = torch.argmin(distance_null_to_mask_test,dim=-1)
                distance_null_to_mask_test_list = distance_null_to_mask_test.cpu().data.numpy().tolist()
                distance_null_to_mask_index_test_list = distance_null_to_mask_index_test.cpu().data.numpy().tolist()

                
                real_pos_hit_test = 0
                virtual_atom_hit = 0   
                masked_tokens_pred_num = torch.sum(masked_tokens_pred,dim=1).cpu().data.numpy().tolist()
                real_pos_cnt = sum(masked_tokens_pred_num)
                virtual_atom_cnt = torch.sum(virtual_index.ne(0)).cpu().data
                virtual_atom_cnt2 = torch.sum(virtual_index.ne(0),dim=-1).cpu().data
                for x, set_t, prob, virtual_atom in zip(masked_tokens_pred_num, distance_null_to_mask_index_test_list, distance_null_to_mask_test_list, virtual_atom_cnt2):
                    virtual_atom = virtual_atom.item()
                    prob_t = [prob[t][set_t[t]] for t in range(virtual_atom) ]
                    set_t = [set_t[t] for t in range(virtual_atom) if prob_t[t] < 0.8 ] 
                    virtual_atom_hit += len(set_t)
                    set_t = set(set_t)
                    for y in range(x):
                        if y in set_t:
                            real_pos_hit_test+=1
                
                real_pos_hit = 0
                real_pos_label_hit = 0
                virtual_atom_hit_pre = 0
                if i==0:
                    virtual_atom_hit_pre = torch.sum(sample[target_key]['virtual_atom_hit_pre'])
                    real_pos_hit = torch.sum(sample[target_key]['real_pos_hit'])   
                    real_pos_label_hit = torch.sum(sample[target_key]['real_pos_label_hit'])
                else:
                    distance_null_to_mask_test = torch.norm((encoder_null_coord.unsqueeze(2) - unmask_coord.unsqueeze(1)), dim=-1)
                    distance_null_to_mask_test.masked_fill_(
                        masked_tokens.unsqueeze(1).to(torch.bool),
                        100000
                    )
                    distance_null_to_mask_test = distance_null_to_mask_test.permute(0, 2, 1)
                    distance_null_to_mask_test.masked_fill_(
                        null_tokens.unsqueeze(1).to(torch.bool),
                        100000,
                    )
                    distance_null_to_mask_test = distance_null_to_mask_test.permute(0, 2, 1)
                    distance_null_to_mask_index_test = torch.argmin(distance_null_to_mask_test,dim=-1)
                    distance_null_to_mask_test_list = distance_null_to_mask_test.cpu().data.numpy().tolist()
                    distance_null_to_mask_index_test_list = distance_null_to_mask_index_test.cpu().data.numpy().tolist()

                    
                    distance_null_to_mask = torch.sum((encoder_null_coord_pre.unsqueeze(2) - unmask_coord.unsqueeze(1)).pow(2),dim=-1)
                    distance_null_to_mask = self.temperature/(distance_null_to_mask+1e-5)
                    distance_null_to_mask.masked_fill_(
                        masked_tokens.unsqueeze(1).to(torch.bool),
                        -100000
                    )
                    prob_null_to_mask = torch.softmax(distance_null_to_mask, dim=-1).detach() 

                    bsz, null_size, unmask_size = prob_null_to_mask.size()
                    mask_null_idx_tgt = torch.argmax(prob_null_to_mask, dim=-1)
                    mask_null_idx_tgt_list = mask_null_idx_tgt.view(bsz, null_size).cpu().data.numpy().tolist()
                    prob_null_to_mask_list = prob_null_to_mask.cpu().data.numpy().tolist()

                    real_pos_hit = 0
                    real_pos_cnt = sum(masked_tokens_pred_num)
                    virtual_atom_hit_pre = 0
                    assert real_pos_cnt > 0
                    assert len(masked_tokens_pred_num) == len(mask_null_idx_tgt_list)
                    virtual_atom_cnt2 = torch.sum(virtual_index.ne(0).type_as(mask_null_idx_tgt),dim=-1).cpu().data
                    for x, set_t, prob, virtual_atom in zip(masked_tokens_pred_num, mask_null_idx_tgt_list, prob_null_to_mask_list, virtual_atom_cnt2):
                        virtual_atom = virtual_atom.item()
                        prob_t = [prob[t][set_t[t]] for t in range(virtual_atom) ]
                        set_t = [set_t[t] for t in range(virtual_atom) if prob_t[t] > 0.4]
                        virtual_atom_hit_pre += len(set_t)
                        set_t = set(set_t)
                        for y in range(x):
                            if y in set_t:
                                real_pos_hit+=1
                    
                    mask_null_idx_tgt = mask_null_idx_tgt.view(bsz, null_size, 1)
                    unmask_coord_repeat = unmask_coord.unsqueeze(1).repeat(1,null_size,1,1)
                    mask_null_idx_tgt_repeat = mask_null_idx_tgt.unsqueeze(-1).repeat(1,1,1,3)
                    coord_null_target = torch.gather(unmask_coord_repeat, 2, mask_null_idx_tgt_repeat).squeeze(2)

                    
                    mask_null_idx_tgt_prob = torch.gather(prob_null_to_mask, 2, mask_null_idx_tgt)
                    coord_null_weight = mask_null_idx_tgt_prob.squeeze(-1).detach()
                    new_tgt_null_prob = torch.cat((1-mask_null_idx_tgt_prob,mask_null_idx_tgt_prob),2)
                    new_tgt_null_init = torch.full( (bsz, null_size, 2), self.null_idx).type_as(unmask_atom)

                    unmask_atom_repeat = unmask_atom.unsqueeze(1).repeat(1,null_size,1)
                    new_tgt_null_init[:,:,1] = torch.gather(unmask_atom_repeat,2, mask_null_idx_tgt).squeeze(-1)
                    new_tgt_null_sample = torch.argmax(new_tgt_null_prob, dim=-1)
                    new_tgt_null_sample = new_tgt_null_sample.view(bsz, null_size, 1)
                    encoder_null_target = torch.gather(new_tgt_null_init, 2, new_tgt_null_sample).squeeze(-1)


                    
                logging_output.update({
                    "sample_size"+'_'+str(i): 1,
                    "bsz": sample[input_key]['src_tokens'].size(0),
                    "seq_len"+'_'+str(i): src_tokens.size(1) * src_tokens.size(0),

                    "real_pos_hit"+'_'+str(i):real_pos_hit,
                    "real_pos_cnt"+'_'+str(i):real_pos_cnt,
                    "real_pos_label_hit"+'_'+str(i):real_pos_label_hit,
                    "real_pos_hit_test"+'_'+str(i):real_pos_hit_test,

                    'virtual_atom_hit_pre'+'_'+str(i):virtual_atom_hit_pre,
                    "virtual_atom_hit"+'_'+str(i):virtual_atom_hit,
                    "virtual_atom_cnt"+'_'+str(i):virtual_atom_cnt
                })
                
                if logits_encoder is not None and encoder_null_target is not None:
                    if null_tokens_pred is not None:
                        logits_encoder_null_post = logits_encoder_null[null_tokens_pred]
                        encoder_null_target_post = encoder_null_target[null_tokens_pred]
                        #     assert (encoder_null_target[xxx][null_tokens_pred[xxx]].ne(self.padding_idx) & encoder_null_target[xxx][null_tokens_pred[xxx]].ne(self.mask_idx ) & encoder_null_target[xxx][null_tokens_pred[xxx]].ne(model.bos_idx) & encoder_null_target[xxx][null_tokens_pred[xxx]].ne(model.eos_idx)).all(), (self.padding_idx, self.mask_idx, model.bos_idx, model.eos_idx, i, encoder_null_target[xxx].cpu().numpy().tolist(),null_tokens_pred[xxx].cpu().numpy().tolist(),xxx, null_tokens_pred.shape, encoder_null_target.shape, merge_method[xxx], unmask_atom.shape, unmask_atom[xxx], mask_null_idx_tgt.shape, mask_null_idx_tgt[xxx], unmask_index.shape, unmask_index[xxx], sample['net_input']['mol_idx'].shape, sample['net_input']['mol_idx'][xxx])
                        assert (encoder_null_target_post.ne(self.padding_idx) & encoder_null_target_post.ne(self.mask_idx ) & encoder_null_target_post.ne(model.bos_idx) & encoder_null_target_post.ne(model.eos_idx)).all(), (encoder_null_target_post, self.padding_idx, self.mask_idx, model.bos_idx, model.eos_idx, i, encoder_null_target.cpu().numpy().tolist(),null_tokens_pred.cpu().numpy().tolist() )
                    null_token_loss_post = F.nll_loss(
                        F.log_softmax(logits_encoder_null_post, dim=-1, dtype=torch.float32),
                        encoder_null_target_post,
                        ignore_index=self.padding_idx,
                        reduction='mean',
                    )

                    null_pred_post = logits_encoder_null_post.argmax(dim=-1)
                    null_hit_post = (null_pred_post == encoder_null_target_post).long().sum()
                    null_cnt_post = null_tokens_pred.long().sum()
                    loss = null_token_loss_post * self.args.masked_loss
                    logging_output["null_token_loss_post"+'_'+str(i)] = null_token_loss_post.data
                    logging_output["null_hit_post"+'_'+str(i)] = null_hit_post.data
                    logging_output["null_cnt_post"+'_'+str(i)] = null_cnt_post


                if encoder_coord is not None and coord_null_target is not None:
                    null_pred_postion_cnt = null_tokens_pred.long().sum()
                    logging_output['null_pred_postion_cnt_'+str(i)] = null_pred_postion_cnt.data
                    if null_pred is not None:
                        null_pred_post = null_pred[null_tokens_pred]
                        encoder_null_coord_post = encoder_null_coord[null_tokens_pred]
                        coord_null_target_post = coord_null_target[null_tokens_pred]
                        distance = torch.norm(encoder_null_coord_post.view(-1, 3) - coord_null_target_post.view(-1, 3),dim=-1)
                        tgt_label = (distance < 2.5).long()
                        null_pred_loss = F.nll_loss(
                            F.log_softmax(null_pred_post, dim=-1, dtype=torch.float32),
                            tgt_label,
                            reduction='mean',
                        ) 
                        loss = loss + null_pred_loss * self.args.null_pred_loss
                        logging_output['null_pred_loss'+'_'+str(i)] = null_pred_loss.data


                        null_pred_position_hit = (null_pred_post.argmax(dim=-1) == tgt_label).long().sum()
                        null_postion_cnt = (tgt_label==0).long().sum()
                        logging_output['null_postion_cnt'] = null_postion_cnt.data
                        logging_output['null_pred_position_hit'] = null_pred_position_hit.data


                    if null_tokens_pred is not None:
                        encoder_null_coord_post = encoder_null_coord[null_tokens_pred]
                        coord_null_target_post = coord_null_target[null_tokens_pred]
                        # 
                        centroid_label_dataset = sample[target_key]['centroid_label_dataset']
                        if centroid_label_dataset.size(0) != encoder_coord[:,0,:].size(0):
                            centroid_label_dataset = centroid_label_dataset.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,centroid_label_dataset.size(1))
                        assert centroid_label_dataset.size(0) == encoder_coord[:,0,:].size(0), (centroid_label_dataset.size(0) , encoder_coord[:,0,:].size(0))
                        encoder_null_coord_post = torch.cat((encoder_coord[:,0,:], encoder_null_coord_post), dim=0)
                        coord_null_target_post = torch.cat((centroid_label_dataset, coord_null_target_post), dim=0)

                    if self.args.weighted_coord_loss > 0:
                        coord_null_loss_post =  F.l1_loss(
                            encoder_null_coord_post.view(-1, 3).float(),
                            coord_null_target_post.view(-1, 3),
                            reduction='none',
                            # reduction='mean',
                        )
                        coord_null_loss_post = torch.mean(coord_null_loss_post * coord_null_weight[null_tokens_pred].view(-1,1))
                    else:
                        l1_clamp_distance = self.args.coord_clamp
                        length_scale = 1
                        if self.args.l1_coord_loss > 0:
                            coord_null_loss_post =  F.l1_loss(
                                encoder_null_coord_post.view(-1, 3).float(),
                                coord_null_target_post.view(-1, 3),
                                # reduction='none',
                                reduction='mean',
                            )
                        else:
                            eps = 1e-5
                            coord_null_loss_post = torch.sqrt(
                                torch.sum((encoder_null_coord_post.view(-1, 3).float() - coord_null_target_post.view(-1, 3)) ** 2, dim=-1) + eps
                            )
                            if l1_clamp_distance is not None:
                                coord_null_loss_post = torch.clamp(coord_null_loss_post, min=0, max=l1_clamp_distance)   
                            coord_null_loss_post = coord_null_loss_post / length_scale
                            coord_null_loss_post = torch.mean(coord_null_loss_post)

                            coord_null_loss_l1 =  F.l1_loss(
                                encoder_null_coord_post.view(-1, 3).float(),
                                coord_null_target_post.view(-1, 3),
                                # reduction='none',
                                reduction='mean',
                            )
                            logging_output['coord_null_loss_l1'+'_'+str(i)] = coord_null_loss_l1.data

                            if i==1:
                                encoder_null_coord_src = masked_coord[:,1:virtual_size+1,:][null_tokens_pred]
                                encoder_null_coord_src = torch.cat((masked_coord[:,0,:], encoder_null_coord_src), dim=0)
                                coord_null_loss_l1_src =  F.l1_loss(
                                    encoder_null_coord_src.view(-1, 3).float(),
                                    coord_null_target_post.view(-1, 3),
                                    # reduction='none',
                                    reduction='mean',
                                )
                                logging_output['coord_null_loss_l1_src'+'_'+str(i)] = coord_null_loss_l1_src.data
                    
                    loss = loss + coord_null_loss_post * self.args.coord_loss
                    logging_output['coord_null_loss_post'+'_'+str(i)] = coord_null_loss_post.data

                    
                if pred_null_dis is not None:
                    pred_null_dis = pred_null_dis[null_tokens_pred]
                    encoder_null_coord_post = encoder_null_coord[null_tokens_pred]
                    coord_null_target_post = coord_null_target[null_tokens_pred]
                    tgt_null_dis_origin = torch.norm(encoder_null_coord_post - coord_null_target_post,dim=-1).detach()
                    tgt_null_dis = (tgt_null_dis_origin/self.args.dist_bin_val).floor().long()
                    tgt_null_dis[tgt_null_dis>self.args.dist_bin-1] = self.args.dist_bin-1

                    pred_null_dis_loss = F.nll_loss(
                        F.log_softmax(pred_null_dis, dim=-1, dtype=torch.float32),
                        tgt_null_dis.view(-1),
                        reduction='mean',
                    )

                    loss = loss + self.args.weighted_distance * pred_null_dis_loss
                    logging_output['pred_null_dis_loss'+'_'+str(i)] = pred_null_dis_loss.data

                    pred_null_dis = torch.sum(torch.softmax(pred_null_dis, dim=-1) * (torch.arange(pred_null_dis.size(1), device=pred_null_dis.device)*self.args.dist_bin_val + 0.5*self.args.dist_bin_val), dim=-1)
                    pred_null_dis_mse_loss = F.mse_loss(
                        pred_null_dis.view(-1).float(),
                        tgt_null_dis_origin.view(-1).float(),
                        reduction="mean",
                    )
                    logging_output['pred_null_dis_mse_loss'+'_'+str(i)] = pred_null_dis_mse_loss.data

                    pred_null_dis_hit = (pred_null_dis[tgt_null_dis_origin<1] < 1).long().sum()
                    pred_null_dis_error = (tgt_null_dis_origin[pred_null_dis<1] > 1).long().sum()
                    logging_output['pred_null_dis_hit'+'_'+str(i)] = pred_null_dis_hit
                    logging_output['pred_null_dis_error'+'_'+str(i)] = pred_null_dis_error
                    logging_output['pred_null_dis_cnt'+'_'+str(i)] = (tgt_null_dis_origin<1).long().sum()
                    logging_output['pred_null_dis_error_cnt'+'_'+str(i)] = (pred_null_dis<1).long().sum()
                
                if pred_merge is not None:
                    pred_merge = pred_merge[:, 1: virtual_size+1, 1: virtual_size+1]
                    
                    merge_label_2 = (encoder_null_target_idx.unsqueeze(-1) - encoder_null_target_idx.unsqueeze(1)).eq(0).long()
                    merge_target = merge_label_2
                    # input_padding_mask = sample[input_key]['src_tokens'][:, 1:virtual_size+1].eq(self.task.dictionary.pad()) | encoder_null_target.eq(self.null_idx)
                    input_padding_mask = null_tokens | encoder_null_target.eq(self.null_idx)
                    merge_target.masked_fill_(
                        input_padding_mask.unsqueeze(1).to(torch.bool),
                        2,
                    )
                    diagonal_mask = torch.eye(merge_target.size(-1), device=merge_target.device).unsqueeze(0).expand(merge_target.size(0),merge_target.size(-1),merge_target.size(-1)).bool()
                    merge_target[diagonal_mask] = 2

                    # null_pos = (pred_null_dis >= self.args.null_dis_range).long()
                    encoder_null_coord_post = encoder_null_coord[null_tokens_pred]
                    coord_null_target_post = coord_null_target[null_tokens_pred]
                    # print(encoder_null_coord.shape, coord_null_target.shape, merge_target.shape)
                    distance = torch.norm(encoder_null_coord - coord_null_target,dim=-1).detach()
                    null_pos = (distance >= self.args.null_dis_range).long()
                    if self.args.null_dist_clip > 0:
                        merge_target.masked_fill_(
                            null_pos.unsqueeze(1).to(torch.bool),
                            0,
                        )
                    else:
                        merge_target.masked_fill_(
                            null_pos.unsqueeze(1).to(torch.bool),
                            2,
                        )

                    assert (src_tokens[:, 1: virtual_size+1].eq(self.task.mask_idx) == null_tokens_pred).all(), (src_tokens[:, 1: virtual_size+1].eq(self.task.mask_idx), null_tokens_pred)
                    input_masking = src_tokens[:, 1: virtual_size+1].eq(self.task.mask_idx) & encoder_null_target.ne(self.null_idx)
                    
                    pred_merge_init = pred_merge.clone().data.cpu().numpy()
                    merge_target_init_gpu = merge_target.clone()
                    merge_target_init = merge_target.clone().data.cpu().numpy()
                    input_masking_init = input_masking.data.cpu().numpy()
                    
                    pred_merge = pred_merge[input_masking]
                    merge_target = merge_target[input_masking]
                    no_pad_pos = merge_target < 2
                    pred_merge = pred_merge[no_pad_pos]
                    merge_target = merge_target[no_pad_pos]
                    pred_merge = pred_merge.float()
                    # print('???', pred_merge)
                    # print('???', merge_target)

                    if self.args.use_focal_loss > 0:
                        # pred_merge = F.log_softmax(pred_merge, dim=-1)
                        pred_merge_prob = torch.cat(((1-pred_merge).unsqueeze(-1), pred_merge.unsqueeze(-1)), dim=-1) #预测的是合并的概率，0是不合并，1是合并，所以1-p被拼在前面
                        pred_merge_prob_log = torch.log(pred_merge_prob)
                        pred_merge_loss =  F.nll_loss(
                            ((1 - pred_merge_prob) ** self.gamma) * pred_merge_prob_log, 
                            merge_target.view(-1), 
                            # weight=self.weight,
                            reduction = 'none'
                        )
                        pred_merge_loss[merge_target==1] = pred_merge_loss[merge_target==1]**self.args.merge_pos_weight
                        pred_merge_loss = torch.mean(pred_merge_loss)
                    else:
                        # pred_merge = torch.softmax(pred_merge, dim=-1)
                        pred_merge_prob = torch.cat(((1-pred_merge).unsqueeze(-1), pred_merge.unsqueeze(-1)), dim=-1)
                        pred_merge_prob_log = torch.log(pred_merge_prob)
                        pred_merge_loss = F.nll_loss(
                            pred_merge_prob_log.float(),
                            merge_target.view(-1),
                            reduction='none',
                        )
                        pred_merge_loss[merge_target==1] = pred_merge_loss[merge_target==1]**self.args.merge_pos_weight
                        pred_merge_loss = torch.mean(pred_merge_loss)
                    loss = loss +  pred_merge_loss * self.args.pred_merge_loss
                    logging_output['pred_merge_loss'+'_'+str(i)] = pred_merge_loss.data

                    pred_merge_post = pred_merge_prob.argmax(dim=-1)
                    pred_merge_hit = (pred_merge_post == merge_target).long().sum()
                    pred_merge_cnt = no_pad_pos.long().sum()

                    logging_output['pred_merge_hit'+'_'+str(i)] = pred_merge_hit
                    logging_output['pred_merge_cnt'+'_'+str(i)] = pred_merge_cnt

                    all_auc = 0
                    for batch_i in range(len(pred_merge_init)):
                        pred_merge_t = pred_merge_init[batch_i][input_masking_init[batch_i],:] 
                        # print('???', merge_target_init[batch_i].shape, input_masking[batch_i].shape)
                        merge_target_t = merge_target_init[batch_i][input_masking_init[batch_i],:] 
                        no_pad_pos_t = merge_target_t < 2
                        pred_merge_t = pred_merge_t[no_pad_pos_t]
                        merge_target_t = merge_target_t[no_pad_pos_t]
                        fpr, tpr, thresholds= sk_metrics.roc_curve(merge_target_t, pred_merge_t, pos_label=1)
                        auc = sk_metrics.auc(fpr, tpr)
                        all_auc+=auc

                    logging_output['pred_merge_auc'+'_'+str(i)] = all_auc/len(pred_merge_init)

                    merge_target = merge_target_init_gpu.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,merge_target_init_gpu.size(1),merge_target_init_gpu.size(2))
                    input_masking = input_masking.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,input_masking.size(1))
                    merge_target = merge_target[input_masking]
                    no_pad_pos = merge_target < 2
                    merge_target = merge_target[no_pad_pos]

                    # merge_label = all_loss[-1][-9] # 
                    merge_label = all_loss[-1][-10]
                    # print('???', merge_label)
                    merge_label = merge_label[:, 1: virtual_size+1, 1: virtual_size+1]
                    # print('???', merge_label)
                    pos_index = merge_target==1
                    print('pos acc: ', (merge_label[input_masking][no_pad_pos][pos_index].view(-1) == merge_target[pos_index]).long().sum()/pos_index.long().sum())
                    neg_index = merge_target==0
                    print('neg acc: ', (merge_label[input_masking][no_pad_pos][neg_index].view(-1) == merge_target[neg_index]).long().sum()/neg_index.long().sum())
                    print('all acc: ', (merge_label[input_masking][no_pad_pos].view(-1) == merge_target).long().sum()/(pred_merge_cnt*repeated_num))
                    


                if self.args.masked_dist_loss > 0 and coord_null_target is not None:
                    def get_masked_dist_loss(masked_distance, masked_distance_target, l1_clamp_distance):
                        tgt_dist_loss = None
                        if self.args.dist_loss == 'mae':
                            masked_dist_loss = F.mae_loss(
                                masked_distance.view(-1).float(),
                                masked_distance_target.view(-1),
                                reduction="mean",
                            )
                        elif self.args.dist_loss == 'mse':
                            masked_dist_loss = F.mse_loss(
                                masked_distance.view(-1).float(),
                                masked_distance_target.view(-1),
                                reduction="mean",
                            )
                        elif self.args.dist_loss == 'smoothl1':
                            masked_dist_loss = F.smooth_l1_loss(
                                masked_distance.view(-1).float(),
                                masked_distance_target.view(-1),
                                reduction="mean",
                                beta=self.args.smoothl1_beta,
                            )
                            tgt_dist_loss = F.smooth_l1_loss(
                                masked_distance.view(-1).float(),
                                masked_distance_target.view(-1),
                                reduction="none",
                                beta=self.args.smoothl1_beta,
                            )
                        elif self.args.dist_loss == 'l1':
                            masked_dist_loss = F.l1_loss(
                                masked_distance.view(-1).float(),
                                masked_distance_target.view(-1),
                                reduction="none",
                            )
                            masked_dist_loss = torch.clamp(masked_dist_loss, min=0, max=l1_clamp_distance) 
                            masked_dist_loss = torch.mean(masked_dist_loss) 
                        else:
                            raise ValueError(
                                "do not implement dist loss function :{}".format(self.args.dist_loss)
                            )
                        with torch.no_grad():
                            if self.args.dist_loss != 'mse':
                                masked_dist_mse_loss = F.mse_loss(
                                masked_distance.view(-1).float(),
                                masked_distance_target.view(-1),
                                reduction="mean",
                                )
                            else:
                                masked_dist_mse_loss = masked_dist_loss
                        return masked_dist_loss, masked_dist_mse_loss, tgt_dist_loss
                    

                    # assert len(virtual_index.ne(0)) == len(src_tokens.eq(self.mask_idx)), (len(virtual_index.ne(0)), len(src_tokens.eq(self.mask_idx)))
                    if self.args.no_dist_head <=0 :
                        masked_distance_null = encoder_pair_rep
                    else:
                        masked_distance_null = torch.norm(encoder_coord.unsqueeze(2) - encoder_coord.unsqueeze(1),dim=-1)

                    input_padding_mask = src_tokens.eq(self.task.dictionary.pad())
                    input_masking = src_tokens.eq(self.task.mask_idx) 
                    input_masking2 = input_masking | src_tokens.eq(model.bos_idx) 
                    dist_target_coord = masked_coord.clone()

                    # print('???', i, len(input_masking), len(null_tokens_pred), dist_target_coord[input_masking].shape, coord_null_target[null_tokens_pred].shape)
                    dist_target_coord[input_masking] = coord_null_target[null_tokens_pred]
                    dist_target_coord[:,0,:] = centroid_label_dataset
                    masked_distance_null_target = torch.norm(dist_target_coord.unsqueeze(2) - dist_target_coord.unsqueeze(1),dim=-1)
                    masked_distance_null_target.masked_fill_(
                        input_padding_mask.unsqueeze(1).to(torch.bool),
                        0,
                    )
                    masked_distance_null = masked_distance_null[input_masking2,:]
                    masked_distance_null_target = masked_distance_null_target[input_masking2,:]
                    non_pad_pos = masked_distance_null_target > 0
                    masked_distance_null = masked_distance_null[non_pad_pos]
                    masked_distance_null_target = masked_distance_null_target[non_pad_pos]
                    masked_dist_null_loss, masked_dist_mse_null_loss, tgt_dist_loss = get_masked_dist_loss(masked_distance_null, masked_distance_null_target, self.args.dist_clamp)
                    loss = loss +  masked_dist_null_loss * self.args.masked_dist_loss #* a
                    logging_output['masked_dist_loss'+'_'+str(i)] = masked_dist_null_loss.data
                    logging_output['masked_dist_mse_loss'+'_'+str(i)] = masked_dist_mse_null_loss.data

                    if self.args.masked_pocket_dist_loss > 0:
                        pocket_coord = sample[input_key]['pocket_coord'].clone()
                        pocket_padding_mask = sample[input_key]['pocket_src_tokens'].eq(self.task.pocket_dictionary.pad())
                        if encoder_coord.size(0)!=pocket_coord.size(0):
                            pocket_coord = pocket_coord.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,pocket_coord.size(1),pocket_coord.size(2))
                            pocket_padding_mask = pocket_padding_mask.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,pocket_padding_mask.size(1))
                        if self.args.no_dist_head <=0 :
                            masked_distance_pocket_null = encoder_pocket_pair_rep
                        else:
                            masked_distance_pocket_null = torch.norm(encoder_coord.unsqueeze(2) - pocket_coord.unsqueeze(1),dim=-1)
                        masked_distance_pocket_null_target = torch.norm(dist_target_coord.unsqueeze(2) - pocket_coord.unsqueeze(1),dim=-1)
                        masked_distance_pocket_null_target.masked_fill_(
                            pocket_padding_mask.unsqueeze(1).to(torch.bool),
                            0,
                        )
                        masked_distance_pocket_null = masked_distance_pocket_null[input_masking2,:]
                        masked_distance_pocket_null_target = masked_distance_pocket_null_target[input_masking2,:]
                        non_pad_pos_t = masked_distance_pocket_null_target > 0
                        masked_distance_pocket_null = masked_distance_pocket_null[non_pad_pos_t]
                        masked_distance_pocket_null_target = masked_distance_pocket_null_target[non_pad_pos_t]
                        masked_pocket_dist_null_loss, masked_pocket_dist_mse_pocket_null_loss, _ = get_masked_dist_loss(masked_distance_pocket_null, masked_distance_pocket_null_target, self.args.pocket_dist_clamp)
                        loss = loss +  masked_pocket_dist_null_loss * self.args.masked_pocket_dist_loss
                        # print('???', masked_pocket_dist_null_loss)
                        logging_output['masked_pocket_dist_loss'+'_'+str(i)] = masked_pocket_dist_null_loss.data
                        logging_output['masked_pocket_dist_mse_loss'+'_'+str(i)] = masked_pocket_dist_mse_pocket_null_loss.data

                    
                if pred_dist_loss is not None and coord_null_target is not None:
                    all_atom_pred_pos = encoder_null_coord.float()
                    all_atom_positions = coord_null_target.float()
                    all_atom_mask = null_tokens_pred.unsqueeze(-1)  # keep dim

                    cutoff = 15.0
                    num_bins = 50
                    eps = 1e-10
                    lddt = self.compute_lddt(
                        all_atom_pred_pos, 
                        all_atom_positions, 
                        all_atom_mask, 
                        cutoff=cutoff, 
                        eps=eps
                    ).detach()

                    bin_index = torch.floor(lddt * num_bins).long()      
                    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
                    lddt_ca_one_hot = torch.nn.functional.one_hot(
                        bin_index, num_classes=num_bins
                    )
                    errors = self.softmax_cross_entropy(pred_dist_loss, lddt_ca_one_hot)
                    all_atom_mask = all_atom_mask.squeeze(-1)
                    pred_dist_loss_loss = self.masked_mean(all_atom_mask, errors, dim=-1, eps=eps)
                    ca_lddt = self.masked_mean(all_atom_mask, lddt, dim=-1, eps=eps)
                    # print('???', pred_dist_loss_loss, ca_lddt)
                    pred_dist_loss_loss = torch.mean(pred_dist_loss_loss)
                    ca_lddt = torch.mean(ca_lddt)

                    pred_dist_loss = torch.softmax(pred_dist_loss, dim=-1)
                    pred_lddt = torch.sum( ( pred_dist_loss * (torch.arange(num_bins, device=pred_dist_loss.device)/num_bins+0.5/num_bins) ), dim=-1)
                    pred_lddt = self.masked_mean(all_atom_mask, pred_lddt, dim=-1, eps=eps)   
                    ouput_for_inference[1].append(pred_lddt)

                    pred_lddt = torch.mean(pred_lddt)

                    loss = loss +  pred_dist_loss_loss * self.args.pred_dist_loss_loss
                    logging_output['pred_dist_loss_loss'+'_'+str(i)] = pred_dist_loss_loss.data
                    logging_output['ca_lddt'+'_'+str(i)] = ca_lddt.data
                    logging_output['pred_lddt'+'_'+str(i)] = pred_lddt.data

                    


                if self.args.dist_regular_loss > 0:

                    if self.args.max_dist > 0:
                        encoder_null_coord_post = encoder_null_coord[null_tokens_pred]
                        encoder_null_coord_prev = pre_coord[null_tokens_pred]

                        encoder_null_coord_post = torch.cat((encoder_coord[:,0,:], encoder_null_coord_post), dim=0)
                        encoder_null_coord_prev = torch.cat((pre_coord_init[:,0,:], encoder_null_coord_prev), dim=0)

                        distance = torch.norm(encoder_null_coord_post.view(-1, 3) - encoder_null_coord_prev.view(-1, 3),dim=-1)
                        distance = self.relu(distance - self.args.max_dist)
                        distance_regular_loss = torch.mean(distance)
                        loss = loss +  distance_regular_loss * self.args.dist_regular_loss
                        logging_output['dist_regular_loss'+'_'+str(i)] = distance_regular_loss.data

        
                if x_norm is not None:
                    loss = loss + self.args.x_norm_loss * x_norm
                    logging_output['x_norm_loss'+'_'+str(i)] = x_norm.data
                

                if delta_pair_repr_norm is not None:
                    loss = loss + self.args.delta_pair_repr_norm_loss * delta_pair_repr_norm
                    logging_output['delta_pair_repr_norm_loss'+'_'+str(i)] = delta_pair_repr_norm.data
                
                if pred_atom_num is not None:
                    pred_atom_num = sample[input_key]['pred_atom_num_mean2']

        
                logging_output['loss'+'_'+str(i)] = loss.data

                if i==0:
                    whole_loss += loss
                else:
                    whole_loss += loss * self.args.reduce_refine_loss

            logging_output['loss'] = whole_loss.data / len(all_loss)

            if inference:
                return whole_loss, 1, logging_output, ouput_for_inference
            else:
                return whole_loss, 1, logging_output
            
        
        
        if inference:
            loss, sample_size, logging_output, ouput_for_inference = inner_forward(inference = True)
            # return loss, sample_size, logging_output, output_atom, output_coord
            return loss, sample_size, logging_output, ouput_for_inference
        else:
            loss, sample_size, logging_output = inner_forward()
            return loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        whole_loss = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size_0", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", whole_loss / sample_size, sample_size, round=3
        )
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)

        # num = len(log.get("sample_size"+'_'+str(i), 0) for log in logging_outputs)

        for i in range(2):
            sample_size = sum(log.get("sample_size"+'_'+str(i), 0) for log in logging_outputs)
            loss_sum = sum(log.get("loss"+'_'+str(i), 0) for log in logging_outputs)
            seq_len = sum(log.get("seq_len"+'_'+str(i), 0) for log in logging_outputs)
            if loss_sum > 0:
                metrics.log_scalar(
                    "loss"+'_'+str(i), loss_sum / sample_size, sample_size, round=3
                )
                metrics.log_scalar(
                    "seq_len"+'_'+str(i), seq_len / bsz, 1, round=3
                )
            
    
            null_token_loss_post = sum(log.get('null_token_loss_post'+'_'+str(i), 0) for log in logging_outputs)
            if null_token_loss_post > 0:
                metrics.log_scalar('null_token_loss_post'+'_'+str(i), null_token_loss_post / sample_size , sample_size, round=3)

                null_acc_post = sum(log.get('null_hit_post'+'_'+str(i), 0) for log in logging_outputs) / sum(log.get('null_cnt_post'+'_'+str(i), 0) for log in logging_outputs)
                metrics.log_scalar('null_acc_post'+'_'+str(i), null_acc_post , sample_size, round=3)
                
            coord_null_loss_post = sum(log.get('coord_null_loss_post'+'_'+str(i), 0) for log in logging_outputs)
            if sample_size > 0:
                metrics.log_scalar('coord_null_loss_post'+'_'+str(i), coord_null_loss_post / sample_size , sample_size, round=3)

            coord_null_loss_l1 = sum(log.get('coord_null_loss_l1'+'_'+str(i), 0) for log in logging_outputs)
            if sample_size > 0:
                metrics.log_scalar('coord_null_loss_l1'+'_'+str(i), coord_null_loss_l1 / sample_size , sample_size, round=3)
            
            coord_null_loss_l1_src = sum(log.get('coord_null_loss_l1_src'+'_'+str(i), 0) for log in logging_outputs)
            if sample_size > 0:
                metrics.log_scalar('coord_null_loss_l1_src'+'_'+str(i), coord_null_loss_l1_src / sample_size , sample_size, round=3)

            real_pos_hit = sum(log.get('real_pos_hit'+'_'+str(i), 0) for log in logging_outputs)
            real_pos_cnt = sum(log.get('real_pos_cnt'+'_'+str(i), 0) for log in logging_outputs)
            if real_pos_cnt > 0:
                real_pos_acc = real_pos_hit/real_pos_cnt
                metrics.log_scalar('real_pos_acc'+'_'+str(i), real_pos_acc , sample_size, round=3)

            
            real_pos_label_hit = sum(log.get('real_pos_label_hit'+'_'+str(i), 0) for log in logging_outputs)
            if real_pos_cnt > 0:
                real_pos_label_hit_acc = real_pos_label_hit/real_pos_cnt
                metrics.log_scalar('real_pos_label_hit_acc'+'_'+str(i), real_pos_label_hit_acc , sample_size, round=3)

            real_pos_hit_test = sum(log.get('real_pos_hit_test'+'_'+str(i), 0) for log in logging_outputs)
            if real_pos_cnt > 0:
                real_pos_test_acc = real_pos_hit_test/real_pos_cnt
                metrics.log_scalar('real_pos_test_acc'+'_'+str(i), real_pos_test_acc , sample_size, round=3)

            virtual_atom_hit = sum(log.get('virtual_atom_hit'+'_'+str(i), 0) for log in logging_outputs)
            virtual_atom_cnt = sum(log.get('virtual_atom_cnt'+'_'+str(i), 0) for log in logging_outputs)
            if virtual_atom_cnt > 0:
                virtual_atom_acc = virtual_atom_hit/virtual_atom_cnt
                metrics.log_scalar('virtual_atom_acc'+'_'+str(i), virtual_atom_acc , sample_size, round=3)

            virtual_atom_hit_pre = sum(log.get('virtual_atom_hit_pre'+'_'+str(i), 0) for log in logging_outputs)
            if virtual_atom_cnt > 0:
                virtual_atom_hit_pre_acc = virtual_atom_hit_pre/virtual_atom_cnt
                metrics.log_scalar('virtual_atom_hit_pre_acc'+'_'+str(i), virtual_atom_hit_pre_acc , sample_size, round=3)

            unmask_pos_hit = sum(log.get('unmask_pos_hit'+'_'+str(i), 0) for log in logging_outputs)
            unmask_pos_cnt = sum(log.get('unmask_pos_cnt'+'_'+str(i), 0) for log in logging_outputs)
            if unmask_pos_cnt > 0:
                unmask_pos_acc = unmask_pos_hit/unmask_pos_cnt
                metrics.log_scalar('unmask_pos_acc'+'_'+str(i), unmask_pos_acc , sample_size, round=3)

            x_norm_loss = sum(log.get('x_norm_loss'+'_'+str(i), 0) for log in logging_outputs)
            if x_norm_loss >0:
                metrics.log_scalar('x_norm_loss'+'_'+str(i), x_norm_loss / sample_size , sample_size, round=3)
            
            delta_pair_repr_norm_loss = sum(log.get('delta_pair_repr_norm_loss'+'_'+str(i), 0) for log in logging_outputs)
            if delta_pair_repr_norm_loss >0:
                metrics.log_scalar('delta_pair_repr_norm_loss'+'_'+str(i), delta_pair_repr_norm_loss / sample_size , sample_size, round=3)

            dist_regular_loss = sum(log.get('dist_regular_loss'+'_'+str(i), 0) for log in logging_outputs)
            if dist_regular_loss > 0:
                metrics.log_scalar('dist_regular_loss'+'_'+str(i), dist_regular_loss / sample_size , sample_size, round=3)

            masked_dist_loss = sum(log.get('masked_dist_loss'+'_'+str(i), 0) for log in logging_outputs)
            if masked_dist_loss > 0:
                metrics.log_scalar('masked_dist_loss'+'_'+str(i), masked_dist_loss / sample_size , sample_size, round=3)

            masked_dist_mse_loss = sum(log.get('masked_dist_mse_loss'+'_'+str(i), 0) for log in logging_outputs)
            if masked_dist_mse_loss > 0:
                metrics.log_scalar('masked_dist_mse_loss'+'_'+str(i), masked_dist_mse_loss / sample_size , sample_size, round=3)
            
            pred_dist_loss_loss = sum(log.get('pred_dist_loss_loss'+'_'+str(i), 0) for log in logging_outputs)
            if pred_dist_loss_loss > 0:
                metrics.log_scalar('pred_dist_loss_loss'+'_'+str(i), pred_dist_loss_loss / sample_size , sample_size, round=3)

            ca_lddt = sum(log.get('ca_lddt'+'_'+str(i), 0) for log in logging_outputs)
            if ca_lddt > 0:
                metrics.log_scalar('ca_lddt'+'_'+str(i), ca_lddt / sample_size , sample_size, round=3)
            pred_lddt = sum(log.get('pred_lddt'+'_'+str(i), 0) for log in logging_outputs)
            if pred_lddt > 0:
                metrics.log_scalar('pred_lddt'+'_'+str(i), pred_lddt / sample_size , sample_size, round=3)
            
            pred_merge_loss = sum(log.get('pred_merge_loss'+'_'+str(i), 0) for log in logging_outputs)
            if pred_merge_loss > 0:
                metrics.log_scalar('pred_merge_loss'+'_'+str(i), pred_merge_loss / sample_size , sample_size, round=3)
            
            pred_merge_hit = sum(log.get('pred_merge_hit'+'_'+str(i), 0) for log in logging_outputs)
            pred_merge_cnt = sum(log.get('pred_merge_cnt'+'_'+str(i), 0) for log in logging_outputs)
            # print('???', pred_merge_hit, pred_merge_cnt)
            if pred_merge_cnt > 0:
                pred_merge_acc = pred_merge_hit/pred_merge_cnt
                metrics.log_scalar('pred_merge_acc'+'_'+str(i), pred_merge_acc , sample_size, round=3)
            
            pred_atom_num_loss = sum(log.get('pred_atom_num_loss'+'_'+str(i), 0) for log in logging_outputs)
            if pred_atom_num_loss > 0:
                metrics.log_scalar('pred_atom_num_loss'+'_'+str(i), pred_atom_num_loss / sample_size , sample_size, round=3)
                
                # pred_atom_hit = sum(log.get('pred_atom_hit'+'_'+str(i), 0) for log in logging_outputs)
                # pred_atom_cnt = sum(log.get('pred_atom_cnt'+'_'+str(i), 0) for log in logging_outputs)
                # pred_atom_acc = pred_atom_hit/pred_atom_cnt
                # metrics.log_scalar('pred_atom_acc'+'_'+str(i), pred_atom_acc , sample_size, round=3)
            
            pred_atom_num_mse_loss = sum(log.get('pred_atom_num_mse_loss'+'_'+str(i), 0) for log in logging_outputs)
            if pred_atom_num_mse_loss > 0:
                metrics.log_scalar('pred_atom_num_mse_loss'+'_'+str(i), pred_atom_num_mse_loss / sample_size , sample_size, round=3)
            
            pred_merge_auc = sum(log.get('pred_merge_auc'+'_'+str(i), 0) for log in logging_outputs)
            if pred_merge_auc > 0:
                metrics.log_scalar('pred_merge_auc'+'_'+str(i), pred_merge_auc / sample_size , sample_size, round=3)
            
            x_norm_loss_pocket = sum(log.get('x_norm_loss_pocket'+'_'+str(i), 0) for log in logging_outputs)
            if x_norm_loss_pocket >0:
                metrics.log_scalar('x_norm_loss_pocket'+'_'+str(i), x_norm_loss_pocket / sample_size , sample_size, round=3)
            
            masked_pocket_dist_loss = sum(log.get('masked_pocket_dist_loss'+'_'+str(i), 0) for log in logging_outputs)
            if masked_pocket_dist_loss > 0:
                metrics.log_scalar('masked_pocket_dist_loss'+'_'+str(i), masked_pocket_dist_loss / sample_size , sample_size, round=3)

            masked_pocket_dist_mse_loss = sum(log.get('masked_pocket_dist_mse_loss'+'_'+str(i), 0) for log in logging_outputs)
            if masked_pocket_dist_mse_loss > 0:
                metrics.log_scalar('masked_pocket_dist_mse_loss'+'_'+str(i), masked_pocket_dist_mse_loss / sample_size , sample_size, round=3)

            null_pred_loss = sum(log.get('null_pred_loss'+'_'+str(i), 0) for log in logging_outputs)
            if null_pred_loss > 0:
                metrics.log_scalar('null_pred_loss'+'_'+str(i), null_pred_loss / sample_size , sample_size, round=3)

            # null_pred_position_hit = sum(log.get('null_pred_position_hit', 0) for log in logging_outputs)
            null_pred_postion_cnt = sum(log.get('null_pred_postion_cnt_'+str(i), 0) for log in logging_outputs)
            # null_postion_cnt = sum(log.get('null_postion_cnt', 0) for log in logging_outputs)
            # if null_pred_postion_cnt > 0 and i==0:
            #     null_pred_position_acc = null_pred_position_hit/null_pred_postion_cnt
                # metrics.log_scalar('null_pred_position_acc', null_pred_position_acc , sample_size, round=3)
                # metrics.log_scalar('null_postion_cnt', null_postion_cnt / bsz , 1, round=3)
            if null_pred_postion_cnt > 0:
                metrics.log_scalar('null_pred_postion_cnt_'+str(i), null_pred_postion_cnt / bsz , 1, round=3)

            pred_null_dis_loss = sum(log.get('pred_null_dis_loss'+'_'+str(i), 0) for log in logging_outputs)
            if pred_null_dis_loss > 0:
                metrics.log_scalar('pred_null_dis_loss'+'_'+str(i), pred_null_dis_loss / sample_size , sample_size, round=3)
                
            pred_null_dis_mse_loss = sum(log.get('pred_null_dis_mse_loss'+'_'+str(i), 0) for log in logging_outputs)
            if pred_null_dis_mse_loss > 0:
                metrics.log_scalar('pred_null_dis_mse_loss'+'_'+str(i), pred_null_dis_mse_loss / sample_size , sample_size, round=3)
            
            pred_null_dis_cnt = sum(log.get('pred_null_dis_cnt'+'_'+str(i), 0) for log in logging_outputs)
            pred_null_dis_hit = sum(log.get('pred_null_dis_hit'+'_'+str(i), 0) for log in logging_outputs)
            pred_null_dis_error = sum(log.get('pred_null_dis_error'+'_'+str(i), 0) for log in logging_outputs)
            pred_null_dis_error_cnt = sum(log.get('pred_null_dis_error_cnt'+'_'+str(i), 0) for log in logging_outputs)
            if pred_null_dis_cnt > 0:
                metrics.log_scalar('null_pred_position_acc'+'_'+str(i), pred_null_dis_hit/pred_null_dis_cnt , sample_size, round=3)
                
            if pred_null_dis_error_cnt > 0:
                metrics.log_scalar('null_pred_position_error'+'_'+str(i), pred_null_dis_error/pred_null_dis_error_cnt , sample_size, round=3)
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
    def compute_lddt(
        self,
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        n = all_atom_mask.shape[-2]
        dmat_true = torch.sqrt(
            eps
            + torch.sum(
                (
                    all_atom_positions[..., None, :]
                    - all_atom_positions[..., None, :, :]
                )
                ** 2,
                dim=-1,
            )
        )
        

        dmat_pred = torch.sqrt(
            eps
            + torch.sum(
                (
                    all_atom_pred_pos[..., None, :]
                    - all_atom_pred_pos[..., None, :, :]
                )
                ** 2,
                dim=-1,
            )
        )
        dists_to_score = (
            (dmat_true < cutoff)
            * all_atom_mask
            * self.permute_final_dims(all_atom_mask, (1, 0))
            * (1.0 - torch.eye(n, device=all_atom_mask.device))
        )

        dist_l1 = torch.abs(dmat_true - dmat_pred)
        

        score = (
            (dist_l1 < 0.5).type(dist_l1.dtype)
            + (dist_l1 < 1.0).type(dist_l1.dtype)
            + (dist_l1 < 2.0).type(dist_l1.dtype)
            + (dist_l1 < 4.0).type(dist_l1.dtype)
        )
        score = score * 0.25

        

        norm = 1.0 / (eps + torch.sum(dists_to_score, dim=-1))
        score = norm * (eps + torch.sum(dists_to_score * score, dim=-1))

        # print('!!!', all_atom_pred_pos.shape, score.shape)

        return score
    def permute_final_dims(self, tensor: torch.Tensor, inds: List[int]):
        zero_index = -1 * len(inds)
        first_inds = list(range(len(tensor.shape[:zero_index])))
        return tensor.permute(first_inds + [zero_index + i for i in inds])
    def masked_mean(self, mask, value, dim, eps=1e-10, keepdim=False):
        mask = mask.expand(*value.shape)
        return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (
            eps + torch.sum(mask, dim=dim, keepdim=keepdim))

    def softmax_cross_entropy(self, logits, labels):
        loss = -1 * torch.sum(
            labels * torch.nn.functional.log_softmax(logits.float(), dim=-1),
            dim=-1,
        )
        return loss