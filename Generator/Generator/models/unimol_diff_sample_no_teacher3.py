import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Callable, Optional, Dict, Tuple, Any, NamedTuple, List
import numpy as np
from torch import Tensor
import math
from unicore.data import Dictionary, data_utils
from multiprocessing import Pool
from tqdm import tqdm
from tqdm import trange
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def merge_heap(merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t, heap_count_t1, heap_count_t2, heap_count):
    c1 = merge_centre_atom[heap_count_t1]
    c2 = merge_centre_atom[heap_count_t2]
    for item in c1:
        merge_centre[item] = heap_count
    for item in c2:
        merge_centre[item] = heap_count
    list_1 = merge_id_list[heap_count_t1]
    list_2 = merge_id_list[heap_count_t2]
    assert len(list_1) > 0
    assert len(list_2) > 0
    assert len(set(list_1)) == len(list_1)
    assert len(set(list_2)) == len(list_2)
    assert len(set(list_1) & set(list_2)) == 0, (list_1, list_2, heap_count_t1, heap_count_t2)
    merge_id_list.append(list_1+list_2)
    merge_id_list[heap_count_t1] = []
    merge_id_list[heap_count_t2] = []
    merge_centre_atom[heap_count] = c1+c2
    for item1 in list_1:
        for item2 in list_2:
            merge_label_res_t[item1][item2] = 1
            merge_label_res_t[item2][item1] = 1
    return merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t

def get_merge_res(value):

    item_index, add_atom, merge_method, real_atom_num, fix_token, pred_atom_num_mean2, merge_label_3, merge_label_5, src_tokens, input_padding_mask, distance, merge_label_6, merge_label, encoder_target, pred_null_dis_value, masked_coord,input_masking, use_real_atom, seed, mask_idx, weighted_distance_temperature, weighted_distance, padding_idx, bos_idx, eos_idx  = value
    if use_real_atom > 0:
        atom_num = real_atom_num.item() - fix_token.item()+add_atom
        used_atom_num = [real_atom_num.item(), 1000, atom_num + fix_token.item()]   
    else:
        atom_num = pred_atom_num_mean2.item() - fix_token.item() + add_atom
        used_atom_num = [real_atom_num.item(), pred_atom_num_mean2.item(), atom_num + fix_token.item()]     
        assert atom_num > 4, (atom_num, pred_atom_num_mean2, add_atom)

    traverse_list = np.arange(len(merge_label_3))
    remain_ = src_tokens.ne(padding_idx) & src_tokens.ne(mask_idx) & src_tokens.ne(bos_idx) & src_tokens.ne(eos_idx)
    remain_num = torch.sum(remain_, dim=-1)
    
    if merge_method ==0:
        merge_id_list = []
        merge_label_t = merge_label_5
        merge_label_res_t = torch.zeros_like(merge_label_t)
        merge_centre = {}
        merge_centre_atom = {}
        high = 0
        heap_count = 0
        atom_count = src_tokens.size(0) - torch.sum(input_padding_mask)

        high_value_list, high_index_list = torch.topk(merge_label_t.reshape(-1), len(merge_label_t.reshape(-1)))

        while high>=0:
            high = high_value_list[0].item()
            if high < 0:
                break
            
            high_index = high_index_list[0].item()
            row_t = high_index // merge_label_t.size(0)
            col_t = high_index % merge_label_t.size(0)
            dist_high = distance[row_t][col_t].item()

            assert col_t != row_t
            if row_t not in merge_centre and col_t not in merge_centre:
                if row_t not in merge_centre and col_t not in merge_centre:
                    merge_centre[row_t] = heap_count
                    merge_centre[col_t] = heap_count
                    # merge_list_index[row_t] = heap_count
                    # merge_list_index[col_t] = heap_count
                    merge_centre_atom[heap_count] = [row_t, col_t]
                    heap_count+=1
                    merge_id_list.append([row_t, col_t])
                    merge_label_res_t[row_t][col_t]=1
                    merge_label_res_t[col_t][row_t]=1
                    atom_count-=1
                else:
                    if row_t not in merge_centre:
                        merge_centre_atom[heap_count] = [row_t]
                        merge_centre[row_t] = heap_count
                        merge_id_list.append([row_t])
                        heap_count+=1

                    else:
                        if col_t not in merge_centre:
                            merge_centre_atom[heap_count] = [col_t]
                            merge_centre[col_t] = heap_count
                            merge_id_list.append([col_t])
                            heap_count+=1
                    
                    heap_count_t1 = merge_centre[row_t]
                    heap_count_t2 = merge_centre[col_t]
                    merge_label_6_t = merge_label_6
                    score_t = merge_label_6_t[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    dis_t = distance[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    if torch.sum(score_t >= high) >=  torch.sum(score_t < high) and (heap_count_t1 != heap_count_t2):
                        merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t = merge_heap(merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t, heap_count_t1, heap_count_t2, heap_count)
                        heap_count+=1
                        atom_count-=1
                
                merge_label_t[row_t][col_t] = -1
                merge_label_t[col_t][row_t] = -1
                
            elif col_t not in merge_centre and row_t in merge_centre:
                if col_t not in merge_centre:
                    merge_centre_atom[heap_count] = [col_t]
                    merge_centre[col_t] = heap_count
                    merge_id_list.append([col_t])
                    heap_count+=1
                    heap_count_t1 = merge_centre[row_t]
                    heap_count_t2 = merge_centre[col_t]
                    merge_label_6_t = merge_label_6
                    score_t = merge_label_6_t[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    dis_t = distance[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    if torch.sum(score_t >= high) >=  torch.sum(score_t < high) and (heap_count_t1 != heap_count_t2):
                        merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t = merge_heap(merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t, heap_count_t1, heap_count_t2, heap_count)
                        heap_count+=1
                        atom_count-=1

                else:
                    heap_count_t1 = merge_centre[row_t]
                    heap_count_t2 = merge_centre[col_t]
                    merge_label_6_t = merge_label_6
                    score_t = merge_label_6_t[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    dis_t = distance[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    if torch.sum(score_t >= high) >=  torch.sum(score_t < high) and (heap_count_t1 != heap_count_t2):
                        merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t = merge_heap(merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t, heap_count_t1, heap_count_t2, heap_count)
                        heap_count+=1
                        atom_count-=1

                merge_label_t[row_t][col_t] = -1
                merge_label_t[col_t][row_t] = -1
            elif row_t not in merge_centre and col_t in merge_centre:
                if row_t not in merge_centre:
                    merge_centre_atom[heap_count] = [row_t]
                    merge_centre[row_t] = heap_count
                    merge_id_list.append([row_t])
                    heap_count+=1
                    heap_count_t1 = merge_centre[col_t]
                    heap_count_t2 = merge_centre[row_t]
                    merge_label_6_t = merge_label_6
                    score_t = merge_label_6_t[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    dis_t = distance[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    if torch.sum(score_t >= high) >=  torch.sum(score_t < high) and (heap_count_t1 != heap_count_t2):
                        merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t = merge_heap(merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t, heap_count_t1, heap_count_t2, heap_count)
                        heap_count+=1
                        atom_count-=1
                else:
                    heap_count_t1 = merge_centre[col_t]
                    heap_count_t2 = merge_centre[row_t]
                    merge_label_6_t = merge_label_6
                    score_t = merge_label_6_t[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    dis_t = distance[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                    if torch.sum(score_t >= high) >=  torch.sum(score_t < high) and (heap_count_t1 != heap_count_t2):
                        merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t = merge_heap(merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t, heap_count_t1, heap_count_t2, heap_count)
                        heap_count+=1
                        atom_count-=1

                merge_label_t[row_t][col_t] = -1
                merge_label_t[col_t][row_t] = -1
            else:
                heap_count_t1 = merge_centre[row_t]
                heap_count_t2 = merge_centre[col_t]
                merge_label_6_t = merge_label_6
                score_t = merge_label_6_t[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                dis_t = distance[merge_id_list[heap_count_t1]][:, merge_id_list[heap_count_t2]]
                if torch.sum(score_t >= high) >=  torch.sum(score_t < high) and (heap_count_t1 != heap_count_t2):
                    
                    merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t = merge_heap(merge_centre_atom, merge_centre, merge_id_list, merge_label_res_t, heap_count_t1, heap_count_t2, heap_count)
                    
                    heap_count+=1
                    atom_count-=1

                merge_label_t[row_t][col_t] = -1
                merge_label_t[col_t][row_t] = -1
            
            high_value_list = high_value_list[1:]
            high_index_list = high_index_list[1:]
            if atom_count <= atom_num:
                break
        print('final count high: ', high, atom_count, atom_num, real_atom_num.item(), add_atom, merge_method)
        used_auc = high
        merge_label_res = merge_label_res_t
    elif merge_method ==1:
        row_index = torch.arange(merge_label_3.size(1))
        merge_label_t = merge_label[input_masking]
        no_pad_pos = merge_label_t < 2
        merge_label_t = merge_label_t[no_pad_pos]
        
        high = max(merge_label_t.float()).item()
        low = min(merge_label_t.float()).item()
        flag = 0
        mid_temp = -1
        count_temp = -1
        with data_utils.numpy_seed(seed[0], seed[1], int(add_atom)):
            select_num = np.random.choice(range(len(merge_label_3) ), 1, replace=False )[0]
        traverse_list = list(np.arange(select_num,len(merge_label_3) ) )+ list(np.arange(select_num))
        while (high-low) > 0.01:
            mid = (high+low)/2
            count=0 
            single_atom_dict = {}
            merge_label_2 = (merge_label_3>=mid).long()
            
            for row_i in traverse_list:
                merge_id = merge_label_2[row_i].ne(0) 
                merge_id_num = torch.sum(merge_id)
                merge_id_repeat = merge_id.unsqueeze(0).repeat(merge_id_num,1)
                if merge_id_num <=0:
                    continue                       
                if merge_id_num == 1 and row_i in single_atom_dict:
                    continue
                merge_id = row_index[merge_id]
                for single_atom in merge_id:
                    single_atom_dict[single_atom.item()] = 1
                count+=1                     
                merge_label_2[:,merge_id] = 0                           
            if count == atom_num:
                flag = 1
                break
            elif count < atom_num: 
                low = mid
            else: 
                high = mid
        print('final count mid: ', mid, count , atom_num,real_atom_num.item(), add_atom, merge_method)

        merge_label_temp = (merge_label >= mid).long()
        merge_label_res = merge_label_temp
        
        merge_label_2 = (merge_label_3>=mid).long()
        merge_id_list = []
        single_atom_dict = {}
        mycount=0
        for row_i in traverse_list:
            merge_id = merge_label_2[row_i].ne(0)
            merge_id_num = torch.sum(merge_id)
            merge_id_repeat = merge_id.unsqueeze(0).repeat(merge_id_num,1)
            if merge_id_num <=0:
                continue
            if merge_id_num == 1 and row_i in single_atom_dict:
                continue
            merge_id = row_index[merge_id]
            for single_atom in merge_id:
                single_atom_dict[single_atom.item()] = 1                      
            merge_id_list.append(merge_id.numpy().tolist())
            merge_label_2[:,merge_id] = 0
            mycount+=1
        assert mycount == count, (mycount, count, mid)
        used_auc = mid

    elif merge_method==2:
        atom_num = int(atom_num)
        merge_id_list = [[] for _ in range(atom_num)]
        kmeans = KMeans(n_clusters=atom_num).fit(masked_coord.numpy()[input_masking,:])
        row_index = np.arange(masked_coord.size(0))[input_masking]
        centers = kmeans.cluster_centers_
        cluster_label = kmeans.labels_
        for i_t in range(len(cluster_label)):
            merge_id_list[cluster_label[i_t]].append(row_index[i_t])
        print('final count: ', atom_num,real_atom_num.item(), add_atom,len(merge_id_list))
        merge_label_res = torch.zeros_like(merge_label)
        for k in range(len(merge_id_list)):
            for i_t in range(len(merge_id_list[k])):
                for j_t in range(i_t):
                    merge_label_res[merge_id_list[k][i_t]][merge_id_list[k][j_t]] =1
                    merge_label_res[merge_id_list[k][j_t]][merge_id_list[k][i_t]] =1
        used_auc = 0


    src_tokens_merge = torch.zeros(src_tokens.size(0)).fill_(padding_idx)
    masked_coord2 = torch.zeros(src_tokens.size(0), 3)
    virtual_index2 = torch.zeros(src_tokens.size(0))
    src_tokens2 = torch.zeros(src_tokens.size(0)).fill_(padding_idx)

    src_tokens2[0] = bos_idx
    src_tokens_merge[0] = bos_idx
    masked_coord2[0] = masked_coord[0]
    
    count = 1
    count2 = 0
    for item in merge_id_list:
        if len(item) == 0:
            continue
        src_tokens2[count] = mask_idx
        atom_type = [encoder_target[x].item() for x in item]

        if weighted_distance > 0:
            dis = pred_null_dis_value[item]
            dis_weight = weighted_distance_temperature/(dis+1e-5)
            dis_softmax = torch.softmax(dis_weight,dim=-1)
            seed_t = int(hash((seed[0], seed[1], int(add_atom), merge_method)) % 1e8)
            state = {"torch_rng_state": torch.get_rng_state()}
            torch.manual_seed(seed_t)
            src_tokens_merge[count] = atom_type[torch.multinomial(dis_softmax, 1).item()]       
            torch.set_rng_state(state["torch_rng_state"])             
            masked_coord2[count] = torch.sum(masked_coord[item] * dis_softmax.unsqueeze(1), dim=0)
            
        else:
            src_tokens_merge[count] = max(atom_type, key=atom_type.count)
            masked_coord2[count] = torch.mean(masked_coord[item], dim=0)
        count += 1
        count2 += 1

    virtual_index2[:count2] = 1
    src_tokens2[count:count+remain_num] = src_tokens[remain_]
    src_tokens_merge[count:count+remain_num] = src_tokens[remain_]
    masked_coord2[count:count+remain_num] = masked_coord[remain_]
    count += remain_num

    new_index_size = count
    new_index_size2 = count2

    return item_index, src_tokens2, src_tokens_merge, masked_coord2, virtual_index2, merge_label_res, new_index_size, new_index_size2, used_auc, used_atom_num


@register_model("unimol_diff_sample_e2e3")
class UniMolDiffSamplee2e3Model(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout", type=float, metavar="D", help="dropout probability for embeddings"
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--contrastive-global-negative", action='store_true', help="use contrastive learning or not"
        )
        parser.add_argument(
            "--auto-regressive", action='store_true', help="use auto regressive generative or not"
        )
        parser.add_argument(
            "--masked-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--contrastive-loss",
            type=float,
            metavar="D",
            help="contrastive loss ratio",
        )
        parser.add_argument(
            "--coord-loss",
            type=float,
            metavar="D",
            help="coord loss ratio",
        )
        parser.add_argument(
            "--dist-loss",
            default='smoothl1',
            choices=['mae','mse','smoothl1','l1'],
            type=str,
            help="coordinate distance loss type",
        )
        parser.add_argument(
            "--dist-regular-loss",
            type=float,
            help="punish those who move a long distance",
        )
        parser.add_argument(
            "--smoothl1-beta",
            default=1.0,
            type=float,
            help="beta in pair distance smoothl1 loss"
        )
        parser.add_argument(
            "--use-gravity-mask",
            type=float,
            help="use gravity mask or not"
        )
        parser.add_argument(
            "--dist-mask",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--weighted-coord-loss",
            type=float,
            help="use weighted or not"
        )
        parser.add_argument(
            "--pred-dist-loss-loss",
            type=float,
            help="use weighted or not"
        )
        parser.add_argument(
            "--only-crossdock", type=float, metavar="L", help="only use cross dock dataset"
        )
        parser.add_argument(
            "--not-use-checkpoint",
            type=bool,
            help="not checkpointing gradient",
        )
        parser.add_argument(
            "--use-initial", type=float, metavar="L", help="do not use vae for initial position"
        )
        parser.add_argument(
            "--refine-type", type=float, metavar="L", help="refine atom type during recycling"
        )
        parser.add_argument(
            "--freeze-param", type=float, metavar="L", help="refine atom type during recycling"
        )
        parser.add_argument(
            "--refine-distance", type=float, metavar="L", help="refine atom type during recycling"
        )
        parser.add_argument(
            "--fix-large-loop", type=float, metavar="L", help="refine atom type during recycling"
        )
        parser.add_argument(
            "--reverse-large-loop", type=float, metavar="L", help="refine atom type during recycling"
        )
        parser.add_argument(
            "--sample-in-loop", type=float, metavar="L", help="refine atom type during recycling"
        )
        parser.add_argument(
            "--pred-merge-loss", type=float, metavar="L", help="refine atom type during recycling"
        )

        parser.add_argument(
            "--sample-atom-type", type=float, metavar="L", help="refine atom type during recycling"
        )
        parser.add_argument(
            "--pred-atom-num",
            type=float,
            default=0.0,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--use-pred-atom-type",
            type=float,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--refine-edge-type",
            type=float,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--atom-num-detach",
            type=float,
            help="mask fragment according to cosine"
        )
        parser.add_argument(
            "--use-pos-tgt",
            type=float,
            help="mask fragment according to cosine",
            default=0.0
        )
        parser.add_argument(
            "--null-dis-range",
            type=float,
            help="mask fragment according to cosine",
            default=2.5
        )
        parser.add_argument(
            "--null-pred-loss",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--weighted-distance",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--weighted-distance-temperature",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--guassian-eps",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--cross-guassian-eps",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--null-dist-clip",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--merge-pos-weight",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--no-dist-head",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--dist-bin-val",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--dist-bin",
            type=int,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--reduce-refine-loss",
            type=float,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--method-num",
            type=int,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--num-recycle0",
            type=int,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--num-recycle1",
            type=int,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--ncpu",
            type=int,
            help="make those distence > k unseen"
        )
        parser.add_argument(
            "--masked-pocket-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.num_types = len(dictionary)
        self.dist_bin = [ x_bin + 1 for x_bin in range(20)]

        self.bos_idx = dictionary.bos()
        self.eos_idx = dictionary.eos()
        self.mask_idx = dictionary.index('[MASK]')
        self.null_idx = dictionary.index('[NULL]')
        self.dictionary = dictionary
        


        self.embed_tokens1 = nn.Embedding(len(dictionary), args.encoder_embed_dim, self.padding_idx)
        self.encoder1 = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            # cross_attention= args.vae_kl_loss > 0
            cross_attention=False,
            pocket_attention=False,
            not_use_checkpoint=args.not_use_checkpoint,
        )

        self.lm_head1 = MaskLMHead(embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
        if self.args.null_pred_loss > 0:
            self.null_head = MaskLMHead(embed_dim=args.encoder_embed_dim,
                output_dim=2,
                activation_fn=args.activation_fn,
                weight=None,
            )
        if args.masked_dist_loss > 0:
            if args.no_dist_head <=0:
                self.dist_head1 = DistanceHead(args.encoder_attention_heads, args.activation_fn)
        if args.pred_atom_num > 0:
            self.atom_num_head = PredictHead(args.encoder_embed_dim, 64, args.activation_fn)
        if self.args.weighted_distance > 0:
            self.null_distance_head = NonLinearHead(args.encoder_embed_dim, args.dist_bin, args.activation_fn)

        self.gbf_proj1 = NonLinearHead(K, args.encoder_attention_heads, args.activation_fn)
        self.gbf1 = GaussianLayer(K, n_edge_type, eps=self.args.guassian_eps)
        self.pair2coord_proj1 = NonLinearHead(args.encoder_attention_heads, 1, args.activation_fn)
        self.pred_merge_head = NonLinearSigmoidHead(args.encoder_attention_heads, 1, args.activation_fn)
        self.classification_heads = nn.ModuleDict()
        self.num_updates = 0


        self.embed_tokens2 = nn.Embedding(len(dictionary), args.encoder_embed_dim, self.padding_idx)
        self.encoder2 = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            # cross_attention= args.vae_kl_loss > 0
            cross_attention=False,
            pocket_attention=False,
            not_use_checkpoint=args.not_use_checkpoint,
        )

        self.lm_head2 = MaskLMHead(embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
        if args.masked_dist_loss > 0:
            if args.no_dist_head <=0:
                self.dist_head2 = DistanceHead(args.encoder_attention_heads, args.activation_fn)

        self.gbf_proj2 = NonLinearHead(K, args.encoder_attention_heads, args.activation_fn)
        self.gbf2 = GaussianLayer(K, n_edge_type, eps=self.args.guassian_eps)
        self.pair2coord_proj2 = NonLinearHead(args.encoder_attention_heads, 1, args.activation_fn)
        self.dist_loss_head = NonLinearHead(args.encoder_embed_dim, 50, args.activation_fn)
        self.apply(init_bert_params)
        
        self.pair2coord_proj1.linear2.weight.data.zero_()
        self.pair2coord_proj1.linear2.bias.data.zero_()
        self.dist_mask = self.args.dist_mask
        self.softmax = nn.Softmax(dim=-1)

        self.pair2coord_proj2.linear2.weight.data.zero_()
        self.pair2coord_proj2.linear2.bias.data.zero_()


        self.encoder = [self.encoder1, self.encoder2]
        self.lm_head = [self.lm_head1, self.lm_head2]
        if args.no_dist_head <=0:
            self.dist_head = [self.dist_head1, self.dist_head2]
        self.gbf_proj = [self.gbf_proj1, self.gbf_proj2]
        self.gbf =[self.gbf1, self.gbf2]
        self.pair2coord_proj = [ self.pair2coord_proj1,  self.pair2coord_proj2]
        self.embed_tokens = [self.embed_tokens1, self.embed_tokens2]




    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        masked_tokens,
        masked_distance,
        masked_coord,
        pred_atom_num_mean2,
        add_atom_num_all,
        edge_type,

        src_tokens2,
        coord_index,
        index_weight,

        all_atom,

        features_only=False,
        no_teacher_forcing=False,
        classification_head_name=None,
        mask_idx=-1,
        inference=False, 
        nsample=1,
        add_atom=0,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        assert masked_tokens==None

        all_loss = []
        src_tokens_merge = None

        init_atom_coo = masked_coord.clone()
        merge_label_res = None
        used_auc = None
        used_atom_num = None
        null_padding = None
        merge_method_all = list(range(self.args.method_num))
        repeated_num = len(merge_method_all) 
        for loop_num in range(2): # 2

            def get_dist_features(dist, et, dist_mask=None):
                n_node = dist.size(-1)
                gbf_feature = self.gbf[loop_num](dist, et)
                gbf_result = self.gbf_proj[loop_num](gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                if dist_mask is not None:
                    graph_attn_bias.masked_fill_(
                        dist_mask.unsqueeze(1).to(torch.bool),
                        float("-inf"),
                    )
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            
            if loop_num > 0:
                bsz = src_tokens_loop0.size(0)
                src_tokens2 = torch.zeros(bsz*repeated_num, src_tokens_loop0.size(1)).fill_(self.padding_idx)
                src_tokens_merge = torch.zeros(bsz*repeated_num, src_tokens_loop0.size(1)).fill_(self.padding_idx)
                masked_coord2 = torch.zeros(bsz*repeated_num, src_tokens_loop0.size(1), 3)
                virtual_index2 = torch.zeros(bsz*repeated_num, src_tokens_loop0.size(1))
                merge_label_res = torch.zeros(bsz*repeated_num, merge_label_5.size(1), merge_label_5.size(2))

                masked_coord = masked_coord_loop0
                src_tokens = src_tokens_loop0
                
                src_tokens2[:,0] = self.bos_idx
                src_tokens_merge[:,0] = self.bos_idx
                masked_coord_cls = masked_coord[:,0]
                masked_coord_cls = masked_coord_cls.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,3)
                masked_coord2[:,0] = masked_coord_cls

                
                new_index_size = 0
                new_index_size2 = 0
                old_batch_list = []
                add_atom_list = []
                merge_method_list = []
                used_auc = [0]*(bsz* repeated_num)
                used_atom_num = [0]*(bsz*repeated_num)
                for temp1 in range(bsz):
                    for temp3 in merge_method_all:
                        old_batch_list.append(temp1)
                        add_atom_list.append(add_atom_num_all[temp1].item())
                        merge_method_list.append(temp3)
                
                new_batch_list = list(range(bsz*repeated_num))
                batch_list = zip(old_batch_list, new_batch_list, add_atom_list, merge_method_list)
                item_index_input = new_batch_list
                add_atom_input  = add_atom_list
                merge_method_input  = merge_method_list
                real_atom_num_input  = real_atom_num.unsqueeze(1).repeat(1,repeated_num).reshape(-1)
                fix_token_input  = fix_token.unsqueeze(1).repeat(1,repeated_num).reshape(-1)
                pred_atom_num_mean2_input  = pred_atom_num_mean2.unsqueeze(1).repeat(1,repeated_num).reshape(-1)
                
                merge_label_3_input  = merge_label_3.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,merge_label_3.size(1),merge_label_3.size(2))
                merge_label_input  = merge_label.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,merge_label.size(1),merge_label.size(2))
                merge_label_5_input  = merge_label_5.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,merge_label_5.size(1),merge_label_5.size(2))
                merge_label_6_input  = merge_label_6.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,merge_label_6.size(1),merge_label_6.size(2))
                
                src_tokens_input  = src_tokens.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,src_tokens.size(1))
                input_padding_mask_input  = input_padding_mask.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,input_padding_mask.size(1))
                distance_input  = distance.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,distance.size(1),distance.size(2))
                encoder_target_input  = encoder_target.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,encoder_target.size(1))
                pred_null_dis_value_input  = pred_null_dis_value.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,pred_null_dis_value.size(1))
                masked_coord_input  = masked_coord.unsqueeze(1).repeat(1,repeated_num,1,1).reshape(-1,masked_coord.size(1),masked_coord.size(2))
                use_real_atom = [self.args.use_real_atom]*len(new_batch_list)
                seed = [(self.args.seed, tt) for tt in range(len(new_batch_list))]
                mask_idx = [self.mask_idx]*len(new_batch_list)
                weighted_distance_temperature = [self.args.weighted_distance_temperature]*len(new_batch_list)
                weighted_distance = [self.args.weighted_distance]*len(new_batch_list)
                padding_idx = [self.padding_idx]*len(new_batch_list)
                input_masking = input_masking.unsqueeze(1).repeat(1,repeated_num,1).reshape(-1,src_tokens.size(1))
                bos_idx = [self.bos_idx]*len(new_batch_list)
                eos_idx = [self.eos_idx]*len(new_batch_list)

                input_list = zip(item_index_input,add_atom_input, merge_method_input, real_atom_num_input, fix_token_input, pred_atom_num_mean2_input, merge_label_3_input, merge_label_5_input, src_tokens_input, input_padding_mask_input, distance_input, merge_label_6_input, merge_label_input, encoder_target_input, pred_null_dis_value_input, masked_coord_input, input_masking, use_real_atom, seed, mask_idx, weighted_distance_temperature, weighted_distance, padding_idx, bos_idx, eos_idx )
                
                with Pool(self.args.ncpu) as pool:
                    item_count = 0
                    for inner_output in tqdm(pool.imap(get_merge_res, input_list ) , total=len(new_batch_list)):
                        # assert 1==0
                        if inner_output ==None:
                            pass
                        else:
                            item_index, src_tokens2_t, src_tokens_merge_t, masked_coord2_t, virtual_index2_t, merge_label_res_t, new_index_size_t, new_index_size2_t, used_auc_t, used_atom_num_t = inner_output
                            
                            src_tokens2[item_index] = src_tokens2_t
                            src_tokens_merge[item_index] = src_tokens_merge_t
                            masked_coord2[item_index] = masked_coord2_t
                            virtual_index2[item_index] = virtual_index2_t
                            merge_label_res[item_index] = merge_label_res_t
                            used_auc[item_index] = used_auc_t
                            used_atom_num[item_index] = used_atom_num_t
                            new_index_size = max(new_index_size, new_index_size_t)
                            new_index_size2 = max(new_index_size2, new_index_size2_t)
                        item_count += 1
                
                new_index_size = math.ceil(new_index_size/8) * 8 
                new_index_size2 = math.ceil(new_index_size2/8) * 8
                src_tokens2 = src_tokens2[:, :new_index_size]
                src_tokens_merge = src_tokens_merge[:, :new_index_size]
                masked_coord2  = masked_coord2[:, :new_index_size,:]
                virtual_index2 = virtual_index2[:,:new_index_size2]

                src_tokens2 = src_tokens2.cuda(torch.distributed.get_rank()).long()
                src_tokens_merge = src_tokens_merge.cuda(torch.distributed.get_rank()).long()
                masked_coord2 = masked_coord2.cuda(torch.distributed.get_rank())
                virtual_index2 = virtual_index2.cuda(torch.distributed.get_rank())
                merge_label_res = merge_label_res.cuda(torch.distributed.get_rank())

                src_tokens = src_tokens2
                masked_coord = masked_coord2

                masked_distance = (masked_coord.unsqueeze(1) - masked_coord.unsqueeze(2)).norm(dim=-1)
                edge_type = src_tokens.unsqueeze(-1) * self.num_types + src_tokens.unsqueeze(1)
                
            init_coord = masked_coord.clone()
            pre_coord = masked_coord.clone()

            input_coords = masked_coord.clone().detach()
            input_src_tokens = src_tokens.clone().detach()

            padding_mask = src_tokens.eq(self.padding_idx)
            if not padding_mask.any():
                padding_mask = None

            virtual_tokens = src_tokens.eq(self.mask_idx) # torch.Size([64, 280])
            assert virtual_tokens.any(), src_tokens
                        
            with utils.torch_seed(self.args.seed, self.num_updates, loop_num):
                if self.training:
                    num_recycle = int((torch.rand(1)*4).data)
                else:
                    num_recycle = 3
            
            if not self.training and loop_num==1:
                num_recycle = self.args.num_recycle1
            if not self.training and loop_num==0:
                num_recycle = self.args.num_recycle0 # 3
            
            if src_tokens_merge is not None:
                src_tokens_recycle = src_tokens_merge
            else:
                src_tokens_recycle = src_tokens

            
            for recycle in range(num_recycle):
                with torch.no_grad():
                    x = self.embed_tokens[loop_num](src_tokens_recycle) #torch.Size([64, 280, 512])
                    graph_attn_bias = get_dist_features(masked_distance, edge_type)
                    encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, _, _ = self.encoder[loop_num](x, padding_mask=padding_mask,\
                    attn_mask=graph_attn_bias, pocket_out=None, \
                    pocket_attn_bias=None, pocket_padding_mask=None, use_checkpoint=False)                                           
                    encoder_distance = None
                    encoder_coord = None
                        
                    if not features_only:
                        if self.args.coord_loss > 0:
                            coords_emb = masked_coord #orch.Size([64, 280, 3])
                            if padding_mask is not None:
                                atom_num = (torch.sum(1-padding_mask.type_as(x),dim=1) - 1).view(-1, 1, 1, 1)
                            else:
                                atom_num = masked_coord.shape[1] - 1
                            delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                            attn_probs = self.pair2coord_proj[loop_num](delta_encoder_pair_rep)
                            coord_update = delta_pos / atom_num * attn_probs
                            coord_update = torch.sum(coord_update, dim=2)
                            encoder_coord = coords_emb + coord_update * 10
                            masked_coord[virtual_tokens] = encoder_coord[virtual_tokens]
                            masked_coord[:,0,:] = encoder_coord[:,0,:]
                            masked_distance = (masked_coord.unsqueeze(1) - masked_coord.unsqueeze(2)).norm(dim=-1)
                            pre_coord = masked_coord.clone()
                            if self.args.refine_type > 0:
                                logits = self.lm_head[loop_num](encoder_rep, masked_tokens)
                                atom_type_sample = self.softmax(logits) #torch.Size([64, 280, 32])
                                if self.args.sample_atom_type > 0:
                                    with utils.torch_seed(self.args.seed, self.num_updates, loop_num, recycle):
                                        atom_type_sample = torch.multinomial(atom_type_sample.view(-1,logits.size(-1)), 1)
                                else:
                                    atom_type_sample = torch.argmax(atom_type_sample, dim=-1)
                                atom_type_sample = atom_type_sample.view(logits.size(0),logits.size(1))#  torch.Size([64, 280])
                                src_tokens_recycle = src_tokens_recycle.clone().detach()
                                src_tokens_recycle[virtual_tokens] = atom_type_sample[virtual_tokens].detach()
                                if self.args.refine_edge_type > 0:
                                    edge_type = src_tokens_recycle.unsqueeze(-1) * self.num_types + src_tokens_recycle.unsqueeze(1)
                                if recycle == 0 and loop_num == 0:
                                    encoder_loop0_0 = encoder_rep.clone().detach()
                                if recycle == 0 and loop_num == 1:
                                    encoder_loop1_0 = encoder_rep.clone().detach()

            x = self.embed_tokens[loop_num](src_tokens_recycle)             
            graph_attn_bias = get_dist_features(masked_distance, edge_type)
            encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_pair_repr_norm = self.encoder[loop_num](x, \
            padding_mask=padding_mask, attn_mask=graph_attn_bias, pocket_out=None, pocket_attn_bias=None, pocket_padding_mask=None)                                                  
            encoder_pair_rep[encoder_pair_rep == float('-inf')] = 0

            encoder_coord = None
            masked_coord2 = init_coord.detach()

            null_pred = None
            if loop_num ==0 and self.args.null_pred_loss > 0:
                null_pred = self.null_head(encoder_rep)

            if not features_only:
                logits = self.lm_head[loop_num](encoder_rep, masked_tokens)
                if self.args.coord_loss > 0:
                    coords_emb = masked_coord 
                    if padding_mask is not None:
                        atom_num = (torch.sum(1-padding_mask.type_as(x),dim=1) - 1).view(-1, 1, 1, 1)
                    else:
                        atom_num = masked_coord.shape[1] - 1
                    delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                    attn_probs = self.pair2coord_proj[loop_num](delta_encoder_pair_rep)
                    coord_update = delta_pos / atom_num * attn_probs
                    coord_update = torch.sum(coord_update, dim=2)
                    encoder_coord = coords_emb + coord_update * 10
                    masked_coord[virtual_tokens] = encoder_coord[virtual_tokens]
                    masked_coord[:,0,:] = encoder_coord[:,0,:]
                    
            
            pred_dist_loss = None
            if loop_num==1:
                pred_dist_loss = self.dist_loss_head(encoder_rep)
            
            if loop_num ==0:
                pred_merge = self.pred_merge_head(encoder_pair_rep) 

            if self.args.masked_dist_loss > 0:
                
                if self.args.no_dist_head <=0:
                    encoder_pair_rep = self.dist_head[loop_num](encoder_pair_rep)
                else:
                    encoder_pair_rep = None

                encoder_pocket_pair_rep = None

            pred_atom_num = None
            
            if self.args.weighted_distance > 0 and loop_num==0:
                pred_null_dis = self.null_distance_head(encoder_rep) 
            
            if loop_num == 1:
                null_pred = None
            
            encoder_coord = masked_coord

            if loop_num ==0:
                masked_coord_loop0 = masked_coord
                src_tokens_loop0 = src_tokens
                if self.args.null_pred_loss > 0:
                    null_type_prob = self.softmax(null_pred)
                    with utils.torch_seed(self.args.seed, self.num_updates):
                        null_type = torch.multinomial(null_type_prob.view(-1,2), 1)
                        null_type_prob = torch.gather(null_type_prob.view(-1,2), 1, null_type)
                        null_type = null_type.reshape(logits.size(0),logits.size(1))
                        null_type_prob = null_type_prob.reshape(logits.size(0),logits.size(1))
                    null_type[:,0] = 1
                    null_type[src_tokens.ne(self.mask_idx) & src_tokens.ne(self.padding_idx)] = 1
                elif self.args.weighted_distance > 0:
                    pred_null_dis_value = torch.sum(torch.softmax(pred_null_dis, dim=-1) * (torch.arange(pred_null_dis.size(-1), device=pred_null_dis.device)*self.args.dist_bin_val + 0.5*self.args.dist_bin_val), dim=-1)
                    null_type = (pred_null_dis_value < self.args.null_dis_range).long()
                    null_type[:,0] = 1
                    null_type[src_tokens.ne(self.mask_idx) & src_tokens.ne(self.padding_idx)] = 1

                    
                real_atom_num = torch.sum(all_atom.ne(self.padding_idx), dim=-1).cpu()

                if self.args.sample_atom_type > 0:
                    pred_src_token = self.softmax(logits)
                    with utils.torch_seed(self.args.seed, self.num_updates, torch.distributed.get_rank()):
                        encoder_target = torch.multinomial(pred_src_token.view(-1,logits.size(-1)), 1)
                        encoder_target = encoder_target.view(logits.size(0),logits.size(1))
                else:
                    encoder_target = torch.argmax(logits, dim=-1)
                encoder_target[null_type.eq(0)] = self.null_idx
                input_padding_mask = src_tokens.ne(self.mask_idx) | encoder_target.eq(self.null_idx)
                diagonal_mask = torch.eye(pred_merge.size(-1), device=pred_merge.device).unsqueeze(0).expand(pred_merge.size(0),pred_merge.size(-1),pred_merge.size(-1)).bool()
                distance = (masked_coord.unsqueeze(1) - masked_coord.unsqueeze(2)).norm(dim=-1)
                distance = 1 / (distance + 1e-4)
                
                merge_label = pred_merge.clone()
                merge_label.masked_fill_(
                    input_padding_mask.unsqueeze(1).to(torch.bool),
                    2,
                )
                merge_label = merge_label.permute(0, 2, 1)
                merge_label.masked_fill_(
                    input_padding_mask.unsqueeze(1).to(torch.bool),
                    2,
                )
                merge_label = merge_label.permute(0, 2, 1)
                merge_label[diagonal_mask] = 2
                input_masking = src_tokens.eq(self.mask_idx) & encoder_target.ne(self.null_idx)
                fix_token = torch.sum(src_tokens.ne(self.mask_idx) & src_tokens.ne(self.bos_idx) & src_tokens.ne(self.padding_idx), dim=-1).cpu()
                merge_atom_num = torch.sum(src_tokens.eq(self.padding_idx), dim=-1)
                merge_label_res = None
                
                merge_label_3 = pred_merge.clone()
                merge_label_3[diagonal_mask] = 100
                merge_label_3.masked_fill_(
                    input_padding_mask.unsqueeze(1).to(torch.bool),
                    0,
                )
                merge_label_3 = merge_label_3.permute(0, 2, 1)
                merge_label_3.masked_fill_(
                    input_padding_mask.unsqueeze(1).to(torch.bool),
                    0,
                )
                merge_label_3 = merge_label_3.permute(0, 2, 1)
                merge_label_3 = merge_label_3.cpu()

                
                merge_label_5 = pred_merge.clone()
                merge_label_5[diagonal_mask] = -1
                merge_label_5.masked_fill_(
                    input_padding_mask.unsqueeze(1).to(torch.bool),
                    -1,
                )
                merge_label_5 = merge_label_5.permute(0, 2, 1)
                merge_label_5.masked_fill_(
                    input_padding_mask.unsqueeze(1).to(torch.bool),
                    -1,
                )
                merge_label_5 = merge_label_5.permute(0, 2, 1)
                merge_label_5 = merge_label_5.cpu()
                merge_label_6 = merge_label_5.clone()

                masked_coord_loop0 = masked_coord_loop0.cpu()
                src_tokens_loop0 = src_tokens_loop0.cpu()
                pred_atom_num_mean2 = pred_atom_num_mean2.cpu()
                encoder_target = encoder_target.cpu()
                input_padding_mask = input_padding_mask.cpu()
                distance = distance.cpu()
                pred_null_dis_value = pred_null_dis_value.cpu()
                
                merge_label = merge_label.cpu()
                input_masking = input_masking.cpu()


            if loop_num==0:
                all_loss.append([logits, encoder_coord, masked_coord2, x_norm, delta_pair_repr_norm, encoder_pair_rep, pre_coord, pred_dist_loss, input_src_tokens, input_coords, pred_merge, pred_atom_num, encoder_pocket_pair_rep, src_tokens_merge, None, None, used_atom_num, used_auc, null_pred, pred_null_dis, None, None, src_tokens, init_atom_coo, encoder_loop0_0])
            else:
                all_loss.append([logits, encoder_coord, masked_coord2, x_norm, delta_pair_repr_norm, encoder_pair_rep, pre_coord, pred_dist_loss, input_src_tokens, input_coords, None, pred_atom_num, encoder_pocket_pair_rep, src_tokens_merge, virtual_index2, merge_label_res, used_atom_num, used_auc, null_pred, None, add_atom_input, merge_method_input, src_tokens, init_atom_coo, encoder_rep])
        
        
        return all_loss

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
    

    
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class ContrastHead(nn.Module):

    def __init__(self, input_dim, output_dim, activation_fn):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(output_dim)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        return x

class PredictHead(nn.Module):

    def __init__(self, input_dim, output_dim, activation_fn):
        super().__init__()
        self.layer_norm = LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x=self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class NonLinearHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden

        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x=self.linear1(x)
        x=self.activation_fn(x)
        x=self.linear2(x)
        return x

class NonLinearSigmoidHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x=self.linear1(x)
        x=self.activation_fn(x)
        x=self.layer_norm(x)
        x=self.linear2(x).view(bsz, seq_len, seq_len)
        # print('???', x.shape)
        x = (x + x.transpose(-1, -2)) * 0.5
        # x=self.softmax(x)
        x = self.sigmoid(x)
        return x

class NonLinearSoftmaxHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x=self.linear1(x)
        x=self.activation_fn(x)
        x=self.layer_norm(x)
        x=self.linear2(x)
        x = (x + x.transpose(-2, -3)) * 0.5
        return x

class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len2, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len2)
        if seq_len == seq_len2:
            x = (x + x.transpose(-1, -2)) * 0.5
        return x



def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024, eps=1e-2):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        self.eps = eps

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + self.eps
        # Todo: try to remove gaussian
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

@register_model_architecture("unimol_diff_sample_e2e3", "unimol_diff_sample_e2e3")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.contrastive_global_negative = getattr(args, "contrastive_global_negative", False)
    args.masked_loss = getattr(args, "masked_loss", 1)
    args.dist_regular_loss = getattr(args, "dist_regular_loss", -1.0)
    args.coord_loss = getattr(args, "coord_loss", 1)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.contrastive_loss = getattr(args, "contrastive_loss", 0)
    args.use_gravity_mask = getattr(args, "use_gravity_mask", 0)
    args.dist_mask = getattr(args, "dist_mask", 0)
    args.x_norm_loss = getattr(args, "x_norm_loss", 0.01)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", 0.01)
    args.weighted_coord_loss = getattr(args, "weighted_coord_loss", 0)
    args.only_crossdock = getattr(args, "only_crossdock", 0)
    args.not_use_checkpoint = getattr(args, "not_use_checkpoint", False)
    args.use_initial = getattr(args, "use_initial", 0.0)
    args.refine_type = getattr(args, "refine_type", 0.0)
    args.pred_dist_loss_loss = getattr(args, "pred_dist_loss_loss", 0.01)
    args.reverse_large_loop = getattr(args, "reverse_large_loop", 0)
    args.fix_large_loop = getattr(args, "fix_large_loop", 0)
    args.sample_in_loop = getattr(args, "sample_in_loop", 0)
    args.pred_merge_loss = getattr(args, "pred_merge_loss", 1.0)
    args.pred_atom_num = getattr(args, "pred_atom_num", 0.0)
    args.use_pred_atom_type = getattr(args, "use_pred_atom_type", 0.0)
    args.refine_edge_type = getattr(args, "refine_edge_type", 0.0)
    

    args.atom_num_detach = getattr(args, "atom_num_detach", 1.0)
    args.null_pred_loss = getattr(args, "null_pred_loss", 1.0)
    args.weighted_distance = getattr(args, "weighted_distance", 0.0)
    args.weighted_distance_temperature = getattr(args, "weighted_distance_temperature", 10.0)
    args.guassian_eps = getattr(args, "guassian_eps", 1e-2)
    args.cross_guassian_eps = getattr(args, "cross_guassian_eps", 1e-2)
    args.null_dist_clip = getattr(args, "null_dist_clip", 0)
    args.null_dist_clip = getattr(args, "sample_atom_type", 1)
    args.merge_pos_weight = getattr(args, "merge_pos_weight", 10)
    args.no_dist_head = getattr(args, "no_dist_head", 0)
    args.dist_bin_val = getattr(args, "dist_bin_val", 0.5)
    args.dist_bin = getattr(args, "dist_bin", 8)
    args.reduce_refine_loss = getattr(args, "reduce_refine_loss", 1.0)
    args.method_num = getattr(args, "method_num", 2)
    args.num_recycle0 = getattr(args, "num_recycle0", 3)
    args.num_recycle1 = getattr(args, "num_recycle1", 15)

    args.masked_pocket_dist_loss = getattr(args, "masked_pocket_dist_loss", 0.0)

    args.ncpu = getattr(args, "ncpu", 8)
    
    
    
    
    

@register_model_architecture("unimol_diff_sample_e2e3", "unimol_diff_sample_e2e3_base")
def unimol_base_architecture(args):
    base_architecture(args)