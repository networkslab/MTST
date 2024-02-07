__all__ = ['MTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.MTST_backbone import MTST_backbone as backbone


from layers.PatchTST_layers import series_decomp

from yacs.config import CfgNode as CN


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 pre_norm: bool = False, pe: str = 'zeros', learn_pe: bool = True, pretrain_head: bool = False,
                 head_type='flatten', verbose: bool = False,
                 res_attention: bool = False, #res_attn default false, get from args
                 store_attn: bool = False,  # see line 28
                 **kwargs
                 ):

        super().__init__()

        # add if store attn:
        store_attn = configs.store_attn  # defaul False, active it in command line

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_branches = configs.n_branches
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len_ls
        stride = configs.stride_ls
        # n_heads_ls = configs.n_heads
        # d_model_ls = configs.d_model #sent by cfg

        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # model
        self.decomposition = decomposition


        cfg = CN()

        configs = vars(configs)
        for k in configs.keys():
            cfg[k] = configs[k]

        res_attention = cfg.get('res_attn', False)
        pe = cfg.get('pe', 'zeros')
        learn_pe = cfg.get('no_learn_pe', True)
        # model

        self.model = backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                   patch_len=patch_len, stride=stride,
                                   max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                   n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                   dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                   attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                   store_attn=store_attn,
                                   pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                   padding_patch=padding_patch,
                                   pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin,
                                   affine=affine,
                                   subtract_last=subtract_last, verbose=verbose,
                                   n_branches=n_branches, cfg=cfg, **kwargs)
        self.res_linear = nn.Linear(context_window, target_window)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x, draw_list, attn_ls = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x, draw_list, attn_ls