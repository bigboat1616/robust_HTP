import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
from datetime import datetime

class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class AuxilliaryEncoderST(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderST, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model//2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        seq_len = 21
        half = x.size(3)//2
        x[:,:,:,0:half*2:2] = x[:,:,:,0:half*2:2] + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,:,:,1:half*2:2] = x[:,:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).unsqueeze(0).repeat_interleave(seq_len, dim=0).to(self.device)).unsqueeze(0)
        return self.dropout(x)

class TransMotionTraj(nn.Module):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, 
                 nlayers_global=4, dropout=0.1, activation='relu', output_scale=1, 
                 obs_and_pred=21, num_tokens=47, device='cuda:0'):
        super(TransMotionTraj, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.obs_and_pred = 21
        self.device = device
        
        # 既存のコンポーネント
        self.fc_in_traj = nn.Linear(4, nhid)
        self.fc_out_traj = nn.Linear(nhid, 4)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=21, device=device)

        # Transformer layers
        encoder_layer_local = nn.TransformerEncoderLayer(
            d_model=nhid,
            nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
            activation=activation
        )
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(
            d_model=nhid,
            nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
            activation=activation
        )
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)

    def forward(self, tgt, padding_mask, metamask=None):
        B, in_F, NJ, K = tgt.shape 
        F = self.obs_and_pred 
        J = self.token_num
        N = NJ // J
        
        # データの整形
        pad_idx = np.repeat([in_F - 1], F - in_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)  
        tgt = tgt[:,i_idx]        
        tgt = tgt.reshape(B,F,N,J,K)
    
        # 軌跡データの抽出
        tgt_traj = tgt[:,:,:,0,:2].to(self.device)  # [B, F, N, 2]
        
        # FFTを適用（時間方向）
        freq_domain = torch.fft.fft(tgt_traj, dim=1)  # [B, F, N, 2]
        
        # 実部と虚部を分離して結合
        freq_features = torch.cat([
            freq_domain.real,  # 実部 [B, F, N, 2]
            freq_domain.imag   # 虚部 [B, F, N, 2]
        ], dim=-1)  # -> [B, F, N, 4]
        
        # 特徴抽出（実数値のまま）
        tgt_traj = self.fc_in_traj(freq_features)
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)
        
        # 既存のTransformer処理
        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1)
        
        tgt = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)
        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)
        out_local = out_local * self.output_scale + tgt
        
        out_local = out_local.reshape(F,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)
        out_global = out_global * self.output_scale + out_local
        
        # 出力生成
        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]
        freq_out = self.fc_out_traj(out_primary)  # [F, 4]
        
        # 実部と虚部を分離して複素数に再構成
        real_part = freq_out[..., :2]
        imag_part = freq_out[..., 2:]
        freq_complex = torch.complex(real_part, imag_part)
        
        # IFFTで時間領域に戻す
        out = torch.fft.ifft(freq_complex, dim=0).real
        out = out.transpose(0, 1).reshape(B, F, 1, 2)
        
        return out

def create_model(config, logger):
    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid = config["MODEL"]["dim_hidden"]
    nhead = config["MODEL"]["num_heads"]
    nlayers_local = config["MODEL"]["num_layers_local"]
    nlayers_global = config["MODEL"]["num_layers_global"]
    dim_feedforward = config["MODEL"]["dim_feedforward"]

    logger.info("Creating trajectory-only model.")
    model = TransMotionTraj(
        tok_dim=seq_len,
        nhid=nhid,
        nhead=nhead,
        dim_feedfwd=dim_feedforward,
        nlayers_local=nlayers_local,
        nlayers_global=nlayers_global,
        output_scale=config["MODEL"]["output_scale"],
        obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
        num_tokens=token_num,
        device=config["DEVICE"]
    ).to(config["DEVICE"]).float()

    return model
  