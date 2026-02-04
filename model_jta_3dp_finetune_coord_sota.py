import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
from datetime import datetime
from stgcn import ST_GCN_18



class STGCN18Reconstructor(nn.Module):
    """ST-GCN-18を用いた座標再構成モデル"""

    def __init__(self, in_channels=3, out_channels=3, feature_dim=256):
        super().__init__()

        self.feature_dim = feature_dim
        self.out_channels = out_channels

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, in_channels))

        graph_cfg = {
            'layout': 'jta_3dp_row',
            'strategy': 'distance',
            'max_hop': 1,
            'dilation': 1
        }

        self.encoder = ST_GCN_18(
            in_channels=in_channels,
            feature_dim=feature_dim,
            graph_cfg=graph_cfg,
            edge_importance_weighting=True,
            data_bn=True
        )

        self.coord_decoder = nn.Linear(feature_dim, out_channels)

    def forward(self, x, return_features=False):
        batch_size, seq_len, num_joints, _ = x.shape

        encoded, _ = self.encoder.extract_feature(x.permute(0, 3, 1, 2))
        encoded = encoded.permute(0, 2, 3, 1)

        decoded = self.coord_decoder(encoded.reshape(-1, self.feature_dim))
        reconstructed = decoded.view(batch_size, seq_len, num_joints, self.out_channels)

        if return_features:
            return reconstructed, encoded

        return reconstructed


class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

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
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)

        return output

class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
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

class Learnedpose3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        return self.dropout(x)

class TransMotion3DP(nn.Module):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21, num_tokens=2, device='cuda:0'):
        super(TransMotion3DP, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.joints_pose = 22
        self.obs_and_pred = 21
        self.device = device
        
        # 軌跡用のレイヤー
        self.fc_in_traj = nn.Linear(2, nhid)
        self.fc_out_traj = nn.Linear(nhid, 2)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=21, device=device)
        
        # 3D姿勢用のレイヤー
        self.fc_in_3dpose = nn.Linear(3, nhid)
        self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)


        # ST-GCNレイヤー
        graph_cfg = {'layout': 'jta_3dp_row', 'strategy': 'distance', 'max_hop': 1, 'dilation': 1}
        self.stgcn = ST_GCN_18(in_channels=3, feature_dim=nhid, graph_cfg=graph_cfg, edge_importance_weighting=True, data_bn=True)

        # Transformerレイヤー
        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)

    def forward(self, tgt, padding_mask, metamask=None):
        B, in_F, NJ, K = tgt.shape 
        F = self.obs_and_pred 
        J = self.token_num
        out_F = F - in_F
        N = NJ // J
        
        # パディングの処理
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)  
        tgt = tgt[:,i_idx]        
        tgt = tgt.reshape(B,F,N,J,K)
    
        # マスクの設定
        mask_ratio_traj = 0.0
        mask_ratio = 0.0

        # 軌跡データの処理
        tgt_traj = tgt[:,:,:,0,:2].to(self.device) 
        traj_mask = torch.rand((B,F,N)).float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        # 3D姿勢データの処理
        tgt_3dpose = tgt[:,:,:,1:,:3].to(self.device)  # 3D姿勢データは2番目のトークンから
        joints_3d_mask = torch.rand((B,F,N,self.joints_pose)).float().to(self.device) > mask_ratio
        joints_3d_mask = joints_3d_mask.unsqueeze(4).repeat_interleave(3,dim=-1)
        tgt_3dpose = tgt_3dpose*joints_3d_mask

        # エンコーディング
        tgt_traj = self.fc_in_traj(tgt_traj) 
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N) 

        tgt_3dpose_9frames = tgt_3dpose[:,:9]  # (B, 9, N, 22, 3)
        tgt_3dpose_original = self.fc_in_3dpose(tgt_3dpose_9frames.transpose(2,3).reshape(B,-1,N,3)) # (B, 9, N, 22,64)
        # データの整合性を保つため、明示的にreshapeとpermuteを行う
        BN = B * N
        # (B, 9, N, 22, 3) -> (B, N, 9, 22, 3) にpermuteしてからreshape
        # これにより、同じbatch内の同じpersonのデータが連続して並ぶ
        tgt_3dpose_stgcn_input = tgt_3dpose_9frames.permute(0, 2, 4, 1, 3).contiguous() 
        tgt_3dpose_stgcn_input = tgt_3dpose_stgcn_input.view(BN, 3, 9, 22) 
        # ST-GCNを通す
        tgt_3dpose_stgcn_output = self.stgcn(tgt_3dpose_stgcn_input)  # (B*N, nhid, 9, 22)
        # 既存のフローに合わせてreshape: (B, 198, N, nhid)
        tgt_3dpose_stgcn_output = tgt_3dpose_stgcn_output.view(B, N, self.nhid, 9*22)  
        tgt_3dpose_stgcn_output = tgt_3dpose_stgcn_output.permute(0, 1, 3, 2).contiguous()  # (B, 198, N, nhid)
        tgt_3dpose = tgt_3dpose_stgcn_output + tgt_3dpose_original
        tgt_3dpose = self.pose3d_encoder(tgt_3dpose)


        # パディングマスクの処理
        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1) 
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1) 
  
        # データの整形
        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid) 
        tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_pose, -1, self.nhid) 
        # アブレーションのために3D姿勢データを0にする
        # tgt_3dpose = tgt_3dpose * 0.0
        # 結合
        tgt = torch.cat((tgt_traj, tgt_3dpose), 0) 

        # Transformer処理
        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)
        out_local = out_local * self.output_scale + tgt

        out_local = out_local[:21].reshape(21,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)

        out_global = out_global * self.output_scale + out_local
        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]
        out_primary = self.fc_out_traj(out_primary) 

        out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)

        return out

def create_model(config, logger):
    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid = config["MODEL"]["dim_hidden"]
    nhead = config["MODEL"]["num_heads"]
    nlayers_local = config["MODEL"]["num_layers_local"]
    nlayers_global = config["MODEL"]["num_layers_global"]
    dim_feedforward = config["MODEL"]["dim_feedforward"]

    logger.info("Creating TransMotion3DP model.")
    model = TransMotion3DP(tok_dim=seq_len,
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

    # 事前学習バックボーンを読み込んで凍結
    ckpt_path = config["MODEL"].get("backbone_ckpt")
    if ckpt_path:
        load_and_freeze_backbone_for_transmotion(model, ckpt_path, device=config["DEVICE"])


    return model 

def load_and_freeze_backbone_for_transmotion(model: TransMotion3DP, ckpt_path: str, device='cpu'):
    """
    ckpt の 'encoder_state_dict'（または 'state_dict'）にある
      - 'encoder.*'           → model.stgcn.*
    を読み込んで、両モジュールを凍結（勾配停止＋BNをeval固定）する。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('encoder_state_dict', ckpt.get('state_dict', ckpt))

    # DataParallel対策: 'module.' を剥がす
    def strip_module(k): 
        return k[7:] if k.startswith('module.') else k

    remapped = {}
    for k, v in sd.items():
        k = strip_module(k)
        if k.startswith('encoder.'):
            new_k = 'stgcn.' + k[len('encoder.'):]
            remapped[new_k] = v
            print(f"loaded {k} -> {new_k}")


    # 既存 state_dict に上書き（形状が一致するキーのみ）
    cur = model.state_dict()
    matched = {k: v for k, v in remapped.items() if k in cur and cur[k].shape == v.shape}
    cur.update(matched)
    missing, unexpected = model.load_state_dict(cur, strict=False)

    print(f"[backbone] loaded {len(matched)} tensors "
          f"(missing={len(getattr(missing, 'missing_keys', missing))}, "
          f"unexpected={len(getattr(unexpected, 'unexpected_keys', unexpected))})")

    # ---- 凍結（勾配停止）----
    for p in model.stgcn.parameters():
        p.requires_grad = False 

    # ---- ST-GCN内のBatchNormを推論固定（統計更新停止＆affineも凍結）----
    for m in model.stgcn.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.track_running_stats = True
            if m.affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    # model.train() を呼んでも stgcn の BN が学習モードに戻らないように軽いラッパ（任意だが推奨）
    orig_train = model.train
    def _train_wrapper(mode: bool = True):
        out = orig_train(mode)
        model.stgcn.eval()
        return out
    model.train = _train_wrapper

    print("[backbone] stgcn are FROZEN (incl. BN eval).")

    