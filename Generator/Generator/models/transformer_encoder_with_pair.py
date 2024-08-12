from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.modules import SelfMultiheadAttention, LayerNorm
from torch.utils.checkpoint import checkpoint
from .cross_encoder_layer import TransformerCrossEncoderLayer
from .transformer_encoder_layer import TransformerEncoderLayer

class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        cross_attention = False,
        pocket_attention = False,
        no_final_layer_norm = False,
        no_final_head_layer_norm = False,
        add_pocket_encoder_layers: int = 0,
        not_use_checkpoint = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln and not no_final_layer_norm:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None
        
        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None
        
        self.encoder_layers = encoder_layers
        self.not_use_checkpoint = not_use_checkpoint
        
        
        if not pocket_attention:
            self.layers = nn.ModuleList(
                [
                    TransformerEncoderLayer(
                        embed_dim=self.embed_dim,
                        ffn_embed_dim=ffn_embed_dim,
                        attention_heads=attention_heads,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        post_ln=post_ln,
                        cross_attention=cross_attention,   
                        pocket_attention=pocket_attention,
                    )
                    for t in range(encoder_layers)
                ]
            )
        else:
            self.layers =nn.ModuleList([])
            for t in range(encoder_layers):
                if (t+1) % 4 == 0 or t == encoder_layers - 1 :
                    self.layers.append(
                        TransformerCrossEncoderLayer(
                            embed_dim=self.embed_dim,
                            ffn_embed_dim=ffn_embed_dim,
                            attention_heads=attention_heads,
                            dropout=dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            activation_fn=activation_fn,
                            post_ln=post_ln,
                            cross_attention=False,   
                            pocket_attention=True,
                        )
                    )
                else:
                    self.layers.append(
                        TransformerEncoderLayer(
                            embed_dim=self.embed_dim,
                            ffn_embed_dim=ffn_embed_dim,
                            attention_heads=attention_heads,
                            dropout=dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            activation_fn=activation_fn,
                            post_ln=post_ln,
                            cross_attention=False,   
                            pocket_attention=False,
                        )
                    )


        if cross_attention:
            self.gaussian_attn_bias = nn.Parameter(torch.randn(1))
            if pocket_attention:
                self.pocket_gaussian_attn_bias = nn.Parameter(torch.randn(1))

    def forward(
        self,
        emb,
        attn_mask: Optional[torch.Tensor] = None,
        dist_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        gravity_mask: Optional[torch.Tensor] = None,
        encoder_out:torch.Tensor=None,
        encoder_attn_bias: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,

        pocket_out:torch.Tensor=None,
        pocket_attn_bias: Optional[torch.Tensor] = None,
        pocket_padding_mask: Optional[torch.Tensor] = None,
        use_checkpoint = True,

    ) -> torch.Tensor:
        # if type(emb) is tuple: 
        #     emb, attn_mask, dist_mask, padding_mask, gravity_mask, encoder_out, encoder_attn_bias, encoder_padding_mask, \
        #     pocket_out, pocket_attn_bias, pocket_padding_mask = emb
        
        # if self.not_use_checkpoint:
        use_checkpoint = False
        assert encoder_out is None

        bsz = emb.size(0)
        seq_len = emb.size(1)        
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask
        mask_pos = padding_mask
        pocket_attn_mask = pocket_attn_bias


        


        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf"), gravity_mask=None, dist_mask=None):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                if gravity_mask is not None:
                    attn_mask_pre = attn_mask.clone()
                    attn_mask.masked_fill_(
                        gravity_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                        fill_val,
                    )
                    # add self attn to virtual atom
                    diagonal_mask = torch.eye(attn_mask.size(-1), device=padding_mask.device).unsqueeze(0).unsqueeze(1).expand(attn_mask.size(0),attn_mask.size(1),attn_mask.size(-1),attn_mask.size(-1)).type_as(padding_mask).to(torch.bool)
                    attn_mask[diagonal_mask] = attn_mask_pre[diagonal_mask]
                if dist_mask is not None:
                    attn_mask.masked_fill_(
                        dist_mask.unsqueeze(1).to(torch.bool),
                        fill_val,
                    )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask
        assert attn_mask is not None

        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask, gravity_mask=gravity_mask)
        

        if encoder_out is not None:
            x = torch.cat((x,encoder_out),dim=1)
            attn_mask = torch.cat((attn_mask, self.gaussian_attn_bias.unsqueeze(1).unsqueeze(1).repeat(attn_mask.size(0),1,attn_mask.size(2))),dim=1)
            attn_mask = torch.cat((attn_mask, self.gaussian_attn_bias.unsqueeze(1).unsqueeze(1).repeat(attn_mask.size(0),attn_mask.size(1),1)),dim=2)
            if pocket_attn_bias is not None:
                pocket_attn_bias = torch.cat((pocket_attn_bias, self.pocket_gaussian_attn_bias.unsqueeze(1).unsqueeze(1).repeat(pocket_attn_bias.size(0),1,pocket_attn_bias.size(2))),dim=1)


        for i in range(len(self.layers)):
            if i >= self.encoder_layers:
                if use_checkpoint:
                    assert 1==0
                    x, attn_mask, _, pocket_attn_bias = checkpoint(self.layers[i], x, attn_mask, padding_mask, None, None, None, True, pocket_out, pocket_attn_bias, pocket_padding_mask)
                else:
                    assert 1==0
                    x, attn_mask, _, pocket_attn_bias = self.layers[i](x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True, pocket_out=pocket_out, pocket_attn_bias=pocket_attn_bias, pocket_padding_mask=pocket_padding_mask)
            else:
                
                if use_checkpoint:
                    assert 1==0
                    x, attn_mask, _, pocket_attn_bias = checkpoint(self.layers[i], x, attn_mask, padding_mask, None, None, None, True, None, None, None)
                else:
                    x, attn_mask, _, pocket_attn_bias_update= self.layers[i](x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True, pocket_out=pocket_out, pocket_attn_bias=pocket_attn_bias, pocket_padding_mask=pocket_padding_mask)
                    if pocket_attn_bias_update is not None:
                        pocket_attn_bias = pocket_attn_bias_update

        if encoder_out is not None:
            x = x[:,:-1,:]
            attn_mask = attn_mask[:,:-1,:-1]
            if pocket_out is not None:
                pocket_attn_bias = pocket_attn_bias[:,:-1,:]

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones((x_norm.size(0),x_norm.size(1)),device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)
        
        if self.final_layer_norm !=  None:
            x = self.final_layer_norm(x)
        
        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, fill_val = 0, gravity_mask=gravity_mask, dist_mask=dist_mask)
        
        attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = delta_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm !=None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
        
        if pocket_attn_mask is not None:
            pocket_attn_mask = pocket_attn_bias.view(bsz, -1, pocket_attn_mask.size(-2), pocket_attn_mask.size(-1)).permute(0, 2, 3, 1).contiguous()
            return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm, pocket_attn_mask  
        else:
            return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
