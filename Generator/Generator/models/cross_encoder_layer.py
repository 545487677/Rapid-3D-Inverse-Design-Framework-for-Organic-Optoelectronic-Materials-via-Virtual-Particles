import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from unicore import utils
from torch import nn
from unicore.modules import LayerNorm, SelfMultiheadAttention
from .crossattn import CrossMultiheadAttention

class TransformerCrossEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
        cross_attention = False,
        pocket_attention = False,
        pocket_post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

        # if cross_attention:
        #     self.cross_attn = CrossMultiheadAttention(
        #     self.embed_dim,
        #         attention_heads,
        #         dropout=attention_dropout,
        #     )

        #     # layer norm associated with the self attention layer
        #     self.corss_attn_layer_norm = LayerNorm(self.embed_dim)
        
        if pocket_attention:
            self.pocket_attn = CrossMultiheadAttention(
            self.embed_dim,
                attention_heads,
                dropout=attention_dropout,
            )
            # layer norm associated with the self attention layer
            self.pocket_attn_layer_norm = LayerNorm(self.embed_dim)
            self.pocket_fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
            self.pocket_fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
            self.pocket_final_layer_norm = LayerNorm(self.embed_dim)
            self.pocket_post_ln = post_ln
            self.pocket_alpha_xattn = nn.Parameter(torch.randn(1))
            self.pocket_alpha_dense = nn.Parameter(torch.randn(1))


    def forward(
        self,
        x,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        encoder_out:torch.Tensor=None,
        encoder_attn_bias: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,

        pocket_out:torch.Tensor=None,
        pocket_attn_bias: Optional[torch.Tensor] = None,
        pocket_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        if type(x) is tuple: 
            x, attn_mask, padding_mask, encoder_out, encoder_attn_bias, encoder_padding_mask, \
            return_attn, pocket_out, pocket_attn_bias, pocket_padding_mask = x
            
        assert pocket_out is not None
        assert pocket_attn_bias is not None
        pocket_attn_weights = None
        if pocket_out is not None:
            # _, pocket_attn_weights = self.pocket_attn(
            #     query=x,
            #     key=pocket_out,
            #     value=pocket_out,
            #     key_padding_mask=pocket_padding_mask,
            #     attn_bias=pocket_attn_bias,
            # )
            residual = x
            if not self.pocket_post_ln:
                x = self.pocket_attn_layer_norm(x)
            # print('???',pocket_attn_bias)
            x, pocket_attn_weights = self.pocket_attn(
                query=x,
                key=pocket_out,
                value=pocket_out,
                key_padding_mask=pocket_padding_mask,
                attn_bias=pocket_attn_bias,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + self.pocket_alpha_xattn * x
            if self.pocket_post_ln:
                x = self.pocket_attn_layer_norm(x)
            
            residual = x
            if not self.pocket_post_ln:
                x = self.pocket_final_layer_norm(x)
            x = self.pocket_fc1(x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.pocket_fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + self.pocket_alpha_dense * x
            if self.pocket_post_ln:
                x = self.pocket_final_layer_norm(x)
        


        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs, pocket_attn_weights
                