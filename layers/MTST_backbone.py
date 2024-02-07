__all__ = ['MTST_backbone']


# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from einops import rearrange, reduce, repeat, einsum
from yacs.config import CfgNode as CN
from layers.rel_pe import RelativeSinPE, RelativeFreqPE

# Cell
class MTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=1, n_branches:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None,  pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, res_attention: bool = True,
                 cfg=CN(),
                 **kwargs
                 ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        if isinstance(patch_len, str):
            patch_len= patch_len.split(',')
            patch_len= [int(i) for i in patch_len]
        if isinstance(stride, str):
            stride = stride.split(',')
            stride = [int(i) for i in stride]

        patch_num = [int((context_window - patch_len[j]) / stride[j] + 1) for j in range(n_branches)]
        if padding_patch == 'end':
            patch_num =[p_n + 1 for p_n in patch_num]

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, padding_patch=padding_patch,
                                n_layers=n_layers, n_branches=n_branches, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, cfg=cfg, **kwargs)

        # Head
        self.head_nf = [d_model * p_n for p_n in patch_num] # to be modified
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.head_dropout = head_dropout
        self.individual = individual
        self.target_window = target_window
        self.n_layers = n_layers
        self.n_branches = n_branches


        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.heads = nn.ModuleList()
            for i in range(self.n_branches):
                self.heads.append(Flatten_Head(self.individual, self.n_vars, self.head_nf[i], target_window, head_dropout=head_dropout))


    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        # do patching in each layer
        # ------- Encoder ------
        z, attn = self.backbone(z)                                                      # z: [bs x nvars x seq_len]

        draw_list = [self.heads[i](z[i]) for i in range(len(z))]  # 3 branches of the last layer to diff linear layer
        z = torch.stack(draw_list, dim=-1).sum(dim=-1, keepdim = False)

        # denorm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            draw_list = [self.revin_layer(z_.permute(0,2,1), 'denorm') for z_ in draw_list]
            # draw_list = [z_.permute(0,2,1) for z_ in draw_list]
            z = z.permute(0,2,1)
        return z, draw_list, attn

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.linear(x)
            x = self.dropout(x)
        return x

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, stride, max_seq_len=1024, padding_patch='end',
                 n_layers=1, n_branches=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False,
                 **kwargs):

        super().__init__()

        cfg = kwargs.get('cfg', CN())

        self.n_branches = n_branches
        self.n_layers = n_layers
        self.seq_len = cfg.get('seq_len', 336)
        self.res_attn = cfg.get('res_attn', False)
        self.individual = cfg.get('individual', False)
        self.n_vars = cfg.get('c_in', 7)
        self.head_dropout = cfg.get('head_dropout', 0)
        self.head_nf = d_model * np.sum(patch_num)


        # Encoder
        self.encoder = nn.ModuleList([
        nn.ModuleList([TSTEncoder(q_len=patch_num[j], patch_len=patch_len[j], stride=stride[j], padding_patch=padding_patch,
                                    d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, n_branches=n_branches,
                                  store_attn=store_attn, cfg=cfg) for j in range(n_branches)])
            for i in range(n_layers)])

        self.bottle_neck = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.seq_len , self.head_dropout)

    def forward(self, x) -> Tensor:
        # x: [bs x nvars x seq_len]
        scores = [None for j in range(self.n_branches)]
        input = x
        for i in range(self.n_layers):
            output_ls = []
            attn_ls = []
            for j in range(self.n_branches):
                if self.res_attn:
                    output, attn, scores[j] = self.encoder[i][j](input, scores=scores[j])
                else:
                    output, attn = self.encoder[i][j](input)
                output_ls.append(output.flatten(2))           # output = [bs x nvars x patch_num x d_model]
                attn_ls += attn

            # except the last layer, repreject it to seq_len
            if i < self.n_layers-1:
                input =  torch.cat(output_ls, dim = -1)  # input = [bs x nvars x patch_num*d_model*3]
                input = self.bottle_neck(input)    # input = [bs x nvars x seq_len]

        return output_ls, attn_ls #return only last layer of branches output, len(z) = n_branches


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, patch_len, stride, padding_patch, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                        n_branches=3,
                        pe='zeros', learn_pe=True, cfg=CN()
                 ):
        super().__init__()

        self.n_layers = n_layers
        self.n_branch = n_branches
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            q_len += 1 # q_len is patch number

        # Input encoding
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        # TST encoding
        self.layer = TSTEncoderLayer(q_len, d_model=d_model, n_heads=n_heads, d_k=d_k,
                                            d_v=d_v, d_ff=d_ff, norm=norm,
                                            attn_dropout=attn_dropout, dropout=dropout,
                                            activation=activation, res_attention=res_attention,
                                            pre_norm=pre_norm, store_attn=store_attn, cfg=cfg,
                                            )
        self.res_attention = res_attention

        # Relative PE
        rel_pe = cfg.get('rel_pe', None)
        if rel_pe is not None and (rel_pe.lower() == 'null' or rel_pe.lower() == 'none'): rel_pe = None
        self.rel_pe = False
        if rel_pe is not None:
            if rel_pe == 'rel_sin':
                self.PE = RelativeSinPE(d_model, linear_freq=False, max_len=5000)
                self.pe_fcs = nn.Linear(d_model, n_heads, bias=False)
                self.pe_pre_fc = nn.Linear(d_model, d_model, bias=True)
            else:
                raise NotImplementedError(f'rel_pe_type=[{rel_pe}] is not supported')

            self.rel_pe = True


    def forward(self, x:Tensor, scores:Optional[Tensor]=None,key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        """
        input: [bs * nvars x patch_num x d_model]
        output: [bs * nvars x patch_num x d_model]
        """
        attn_bias=None

        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        x = x.reshape(x.size(0), x.size(1), -1, self.patch_len)

        # x: [bs x nvars x patch_num x patch_len]
        n_vars = x.shape[1]
        # Linear Projection
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
        u = rearrange(x, 'b n p d -> (b n) p d')
        u = self.dropout(u)  # u: [bs * nvars x patch_num x d_model]

        #Transformer
        input = u

        if self.rel_pe:
            # pre-fc of rel_pe
            rel_pe_val, rel_pe_idx = self.PE(input, x_dec=None)
            rel_pe = rel_pe_val[:, torch.relu(rel_pe_idx)] - rel_pe_val[:, torch.relu(-rel_pe_idx)]
            rel_pe = torch.relu(self.pe_pre_fc(rel_pe))
            # expected shape [1 x n_heads x q_len x k_len]
            attn_bias = self.pe_fcs(rel_pe).permute(0, 3, 1, 2)

        if self.res_attention:
            output, attn, scores = self.layer(input, prev=scores, key_padding_mask=key_padding_mask,
                                                  attn_mask=attn_mask, attn_bias=attn_bias)
            output = rearrange(output, '(b n) p d -> b n p d', n=n_vars)
            return output, attn, scores  # List[Tensor( bs x nvars x patch_num x d_model)]
        else:
            output, attn= self.layer(input, key_padding_mask=key_padding_mask,
                                                  attn_mask=attn_mask, attn_bias=attn_bias)
            output = rearrange(output, '(b n) p d -> b n p d', n=n_vars)
            return output, attn  # List[Tensor( bs x nvars x patch_num x d_model)]


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False, cfg=CN(),
                 ):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v


        self.pre_norm = pre_norm

        # Multi-Head attention
        self.res_attention = cfg.get('res_attn', False)
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, cfg=cfg)
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = cfg.pre_norm
        self.store_attn = store_attn




    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                attn_bias:Optional[Tensor]=None) -> Tensor:
        # src : [10272, 85, 128] = [bs x nvar, patch_num, d_model]
        # Multi-Head attention sublayer
        res = src
        if self.pre_norm:
            src = self.norm_attn(src)

        ## Multi-Head attention
        if self.res_attention:
            src, attn, scores = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                       attn_bias=attn_bias)
        else:
            src, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                   attn_bias=attn_bias)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = res + self.dropout_attn(src) # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.norm_attn(src)

        res = src
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)

        ## Position-wise Feed-Forward
        src = self.ff(src)
        ## Add & Norm
        src = res + self.dropout_ffn(src) # Add: residual connection with residual dropout
        if not self.pre_norm: # default pre_norm = False
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, attn, scores

        else:
            return src, attn # always save attn_list



class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,cfg=CN()):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                attn_bias:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_headsn_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                              attn_bias=attn_bias)
        else:
            output, attn_weights = self.attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                 attn_bias=attn_bias)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                attn_bias:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
            attn_bias       : [1 x seql_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        save_score = []
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        save_score.append(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, save_score, attn_scores
        # else: return output, attn_weights
        else: return output, save_score #instead of saving final weight, save weight_list=[qk_score, bias_score, total_score]

