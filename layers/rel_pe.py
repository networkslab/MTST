import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    '''
        Data Embedding for LSTiT
        - value_emb + temporal_emb
        > use relative PE instead of absolute PE
    '''
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, use_abs_pe=True):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

        self.use_abs_pe = use_abs_pe

        if self.use_abs_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.pe_fc = nn.Linear(d_model, d_model)


    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)

        if self.use_abs_pe:
            x = x + self.pe_fc(self.position_embedding(x))

        return self.dropout(x)


class RelativeSinPE(nn.Module):
    '''
        Relative Sine-PE to enable Frequency info as inductive bias
    '''
    def __init__(self, d_model, max_len=5000, linear_freq=False):
        '''
        :param d_model: The dimension of PE
        :param max_len: The maximum length allowed
        :param linear_freq: Use Linear Freq (DFT) instead of Exponential Freq
        '''
        super().__init__()
        # Compute the positional encodings once in log space.
        BASE = 10000.0

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        if linear_freq:
            div_term = (torch.arange(0, d_model, 2).float())/ d_model * BASE
        else:
            div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (L, d_model)

        pe = F.pad(pe, (0, 0, 1, 0), mode="constant", value=0.)
        # pe = torch.cat([torch.zeros_like(pe[:1]), pe], dim=0)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # (1, max_len, d)

    @torch.no_grad()
    def forward(self, x_enc, x_dec=None, overlap_len=0):
        enc_idx = torch.arange(x_enc.size(1))
        enc_enc = enc_idx.unsqueeze(1)  - enc_idx.unsqueeze(0) # (L_in, L_in)
        if x_dec is None:
            return self.pe, enc_enc
        dec_idx = torch.arange(x_dec.size(1)) + x_enc.size(1) - max(overlap_len, 0)
        dec_dec = dec_idx.unsqueeze(1) - dec_idx.unsqueeze(0) # (L_out, L_out)
        dec_enc = dec_idx.unsqueeze(1) - enc_idx.unsqueeze(0) # (L_out, L_in)
        return self.pe, enc_enc, dec_enc, dec_dec



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
    def forward(self, x_enc, x_dec=None, overlap_len=0):
        enc_idx = torch.arange(x_enc.size(1))
        enc_enc = enc_idx.unsqueeze(1)  - enc_idx.unsqueeze(0) # (L_in, L_in)
        if x_dec is None:
            return self.pe, enc_enc

        dec_idx = torch.arange(x_dec.size(1)) + x_enc.size(1) - max(overlap_len, 0)
        dec_dec = dec_idx.unsqueeze(1) - dec_idx.unsqueeze(0) # (L_out, L_out)
        dec_enc = dec_idx.unsqueeze(1) - enc_idx.unsqueeze(0) # (L_out, L_in)

        enc_enc = torch.abs(enc_enc)
        dec_dec = torch.abs(dec_dec)
        dec_enc = torch.abs(dec_enc )

        enc_enc = torch.masked_fill(enc_enc, enc_enc>self.max_len-1, self.max_len)
        dec_dec = torch.masked_fill(dec_dec, dec_dec>self.max_len-1, self.max_len)
        dec_enc = torch.masked_fill(dec_enc, dec_enc>self.max_len-1, self.max_len)

        return self.pe, enc_enc, dec_enc, dec_dec


class SinDegEncoder(nn.Module):
    def __init__(self, hidden_dim=64, constant=10000):
        super().__init__()
        self.eps = 100 # to make the wave smaller to aovid better sensitivity on smaller value
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-np.log(constant)/self.hidden_dim))
        self.register_buffer('div', div)

    def forward(self, batch):

        deg = batch.deg
        deg = deg.flatten(0) * self.eps # [B]
        degenc = deg.unsqueeze(-1) * self.div # auto broadcast: [B, 1] x [D/2] --> [B, D/2]
        degenc = torch.cat([torch.sin(degenc), torch.cos(degenc)], dim=2) # [B, D/2] --> [B, D]

        batch.x = batch.x + self.fc(degenc) if 'x' in batch else self.fc(degenc)

        return batch

