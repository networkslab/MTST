import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math



class RelativeFreqPE(nn.Module):
    '''
        Relative Frequency Based PE
    '''
    def __init__(self, d_pe=128, max_len=2000):
        super().__init__()
        # Compute the positional encodings once in log space.
        BASE = 10000.0
        self.max_len = max_len

        pe = torch.zeros(max_len+1, max_len).float()
        pe.require_grad = False

        position = torch.arange(max_len)
        for freq in range(1, max_len):
            mask = position % freq == 0
            pe[freq][mask] = 1.

        pe[0][0] = 1. # identity identification
        pe[-1] = 0. # out of range indicator
        pe = pe[:, :d_pe]  # take top d_pe as the PE

        self.register_buffer('pe', pe) # (d+1, d)

    @torch.no_grad()
    def forward(self):
        return self.pe



    # def forward(self, x_enc, x_dec=None, overlap_len=0):
    #     enc_idx = torch.arange(x_enc.size(1))
    #     enc_enc = enc_idx.unsqueeze(1)  - enc_idx.unsqueeze(0) # (L_in, L_in)
    #     if x_dec is None:
    #         return self.pe, enc_enc
    #
    #     dec_idx = torch.arange(x_dec.size(1)) + x_enc.size(1) - max(overlap_len, 0)
    #     dec_dec = dec_idx.unsqueeze(1) - dec_idx.unsqueeze(0) # (L_out, L_out)
    #     dec_enc = dec_idx.unsqueeze(1) - enc_idx.unsqueeze(0) # (L_out, L_in)
    #
    #     enc_enc = torch.abs(enc_enc)
    #     dec_dec = torch.abs(dec_dec)
    #     dec_enc = torch.abs(dec_enc )
    #
    #     enc_enc = torch.masked_fill(enc_enc, enc_enc>self.max_len-1, self.max_len)
    #     dec_dec = torch.masked_fill(dec_dec, dec_dec>self.max_len-1, self.max_len)
    #     dec_enc = torch.masked_fill(dec_enc, dec_enc>self.max_len-1, self.max_len)
    #
    #     return self.pe, enc_enc, dec_enc, dec_dec
