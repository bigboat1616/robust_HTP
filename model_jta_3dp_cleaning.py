import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skeleton_mae.graphmae.models.stgcn import ST_GCN_18


class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal: bool = False, need_attn: bool = False):
        # Based on torch.nn.TransformerEncoderLayer.forward, but optionally returns attention.
        # src shape: (S, B, E) when batch_first=False (default)
        x = src

        if self.norm_first:
            x_norm = self.norm1(x)
            attn_out, attn_w = self._sa_block(x_norm, src_mask, src_key_padding_mask, is_causal, need_attn)
            x = x + attn_out
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_out, attn_w = self._sa_block(x, src_mask, src_key_padding_mask, is_causal, need_attn)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self._ff_block(x))

        if need_attn:
            return x, attn_w
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal, need_attn):
        attn_output, attn_output_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_attn,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(attn_output), attn_output_weights


class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            if get_attn:
                output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, need_attn=True)
                attn_matrices.append(attn)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        if get_attn:
            return output, attn_matrices
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
            if get_attn:
                output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, need_attn=True)
                attn_matrices.append(attn)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)

        if get_attn:
            return output, attn_matrices
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

class LearnedIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.person_encoding = nn.Embedding(1000, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:

        seq_len = 21
   
        x = x + self.person_encoding(torch.arange(num_people).repeat_interleave(seq_len, dim=0).to(self.device)).unsqueeze(1)
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
        self.id_encoder = LearnedIDEncoding(nhid, dropout, seq_len=21, device=device)

        
        # 3D姿勢用のレイヤー
        self.fc_in_3dpose = nn.Linear(3, nhid)
        self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)


        # 再構成モデル
        graph_cfg = {'layout': 'jta_3dp_row', 'strategy': 'distance', 'max_hop': 1, 'dilation': 1}
        self.stgcn = ST_GCN_18(in_channels=3, feature_dim=nhid, graph_cfg=graph_cfg, edge_importance_weighting=True, data_bn=True, layer_num=3)

        self.coord_decoder = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 3),
        )

        # Transformerレイヤー
        encoder_layer_local = TransformerEncoderLayerWithAttn(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = TransformerEncoderLayerWithAttn(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)

    def _joint_attention_from_local(self, attn_local, in_F, average_heads=True, average_batch=True):
        if not attn_local:
            return None
        # attn_local[-1]: (B*N, nhead, S, S) where S = traj(21) + pose(in_F*22)
        last_attn = attn_local[-1]
        traj_len = self.obs_and_pred
        pose_len = in_F * self.joints_pose
        pose_start = traj_len
        pose_attn = last_attn[:, :, pose_start:pose_start + pose_len, pose_start:pose_start + pose_len]

        if average_heads:
            pose_attn = pose_attn.mean(dim=1)
        if average_batch:
            pose_attn = pose_attn.mean(dim=0)

        # (in_F, 22, in_F, 22) -> (22, 22) joint-to-joint
        pose_attn_22 = pose_attn.view(in_F, self.joints_pose, in_F, self.joints_pose).mean(dim=(0, 2))
        return pose_attn_22

    def forward(self, tgt, padding_mask, metamask=None, get_attn=False):
        tgt = tgt.to(self.device)
        B, in_F, NJ, K = tgt.shape 
        F = self.obs_and_pred 
        J = self.token_num
        out_F = F - in_F
        N = NJ // J
        # パディングの処理
        i_idx = torch.cat([torch.arange(in_F), torch.full((out_F,), in_F-1)]).to(tgt.device)
        tgt = tgt[:,i_idx]        
        tgt = tgt.reshape(B,F,N,J,K)

        mask_ratio = 0.0

        # 軌跡データの処理
        tgt_traj = tgt[:,:,:,0,:2] 

        # 3D姿勢データの処理
        tgt_3dpose = tgt[:,:,:,1:,:3]  
        # ちょうどK関節をマスク (関節15は除外)
        allowed_joints = torch.tensor(
            [j for j in range(self.joints_pose) if j != 15],
            device=self.device,
        )
        num_mask = int(round(mask_ratio * allowed_joints.numel()))
        num_mask = max(0, min(num_mask, allowed_joints.numel()))
        joints_3d_mask = torch.zeros((B, F, N, self.joints_pose), device=self.device, dtype=torch.bool)
        if num_mask > 0:
            scores = torch.rand((B, F, N, allowed_joints.numel()), device=self.device)
            _, idx = scores.topk(num_mask, dim=-1)
            selected = allowed_joints[idx]
            joints_3d_mask.scatter_(dim=-1, index=selected, value=True)
        mask_token = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=tgt_3dpose.dtype)
        tgt_3dpose = torch.where(joints_3d_mask.unsqueeze(-1), mask_token, tgt_3dpose)

        # tgt_3dpose = tgt_3dpose - tgt_3dpose[:,(in_F-1):in_F,:,:,:]

        # エンコーディング
        tgt_traj = self.fc_in_traj(tgt_traj) 
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N) 

        BN = B * N
        # print("3dpose_row mean/std:", tgt_3dpose.mean().item(), tgt_3dpose.std().item())
        tgt_3dpose = tgt_3dpose[:,:9]  # (B, 9, N, 22, 3)
        pose_input_9 = tgt_3dpose
        tgt_3dpose = tgt_3dpose.permute(0, 2, 4, 1, 3).contiguous()
        tgt_3dpose = tgt_3dpose.reshape(BN, 3, 9, 22)
        # ST-GCNを通す
        tgt_3dpose = self.stgcn(tgt_3dpose)  # (B*N, nhid, 9, 22)
        tgt_3dpose = tgt_3dpose.permute(0, 2, 3, 1)  # (B*N, 9, 22, nhid)
        tgt_3dpose = self.coord_decoder(tgt_3dpose.reshape(-1, self.nhid))  # (B*N*9*22, 3)
        tgt_3dpose = tgt_3dpose.view(B, N, 9, 22, 3)
        recon_pose_9 = tgt_3dpose.permute(0, 2, 1, 3, 4).contiguous()  # (B, 9, N, 22, 3)
        # 3D座標が完全に0の関節のみ再構成で置き換え（関節15は除外）
        zero_mask = (pose_input_9 == 0).all(dim=-1)  # (B, 9, N, 22)
        allowed_joints_mask = torch.ones((1, 1, 1, self.joints_pose), device=self.device, dtype=torch.bool)
        allowed_joints_mask[..., 15] = False
        replace_mask = zero_mask & allowed_joints_mask
        pose_input_9 = torch.where(replace_mask.unsqueeze(-1), recon_pose_9, pose_input_9)
        # print("stgcn mean/std:", tgt_3dpose.mean().item(), tgt_3dpose.std().item())
        # 既存のフローに合わせてreshape: (B, 198, N, nhid)へ変換
        tgt_3dpose = pose_input_9.permute(0, 1, 3, 2, 4).contiguous().reshape(B, 9 * 22, N, 3)
        tgt_3dpose = self.fc_in_3dpose(tgt_3dpose)
        tgt_3dpose = self.pose3d_encoder(tgt_3dpose)

        # パディングマスクの処理
        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1) 
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1) 
  
        # データの整形
        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid) 
        tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_pose, -1, self.nhid) 
        #アブレーションのための3dposeゼロ埋め
        # tgt_3dpose = torch.zeros(in_F*self.joints_pose, B*N , self.nhid).to(self.device)
        # print("traj mean/std:", tgt_traj.mean().item(), tgt_traj.std().item())
        # print("3dpose mean/std:", tgt_3dpose.mean().item(), tgt_3dpose.std().item())
        # 結合
        tgt = torch.cat((tgt_traj, tgt_3dpose), 0) 
        # Transformer処理
        if get_attn:
            out_local, attn_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local, get_attn=True)
        else:
            out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)
        # print("out_local mean/std:", out_local.mean().item(), out_local.std().item())
        out_local = out_local * self.output_scale + tgt

        out_local = out_local[:21].reshape(21,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
        if get_attn:
            out_global, attn_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global, get_attn=True)
        else:
            out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)

        out_global = out_global * self.output_scale + out_local
        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]
        out_primary = self.fc_out_traj(out_primary) 

        out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)

        if get_attn:
            joint_attn_22 = self._joint_attention_from_local(attn_local, in_F)
            return out, {"local": attn_local, "global": attn_global, "joint_attn_22": joint_attn_22}
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

    return model 


def _resolve_state_dict(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("model", ckpt.get("encoder_state_dict",
               ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))))
    return ckpt


def _strip_module_prefix(k):
    return k[7:] if k.startswith("module.") else k


def load_reconstruction_weights(model: TransMotion3DP, ckpt_path: str, device="cpu"):
    """
    再構成モデル(stgcn + coord_decoder)だけを読み込む。
    encoder.* -> stgcn.* も許容。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = _resolve_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(sd)}")

    remapped = {}
    for k, v in sd.items():
        k = _strip_module_prefix(k)
        if k.startswith("encoder."):
            k = "stgcn." + k[len("encoder."):]
        remapped[k] = v

    cur = model.state_dict()
    matched = {
        k: v for k, v in remapped.items()
        if (k.startswith("stgcn.") or k.startswith("coord_decoder."))
        and k in cur and cur[k].shape == v.shape
    }
    cur.update(matched)
    missing, unexpected = model.load_state_dict(cur, strict=False)
    print(f"[recon] loaded {len(matched)} tensors "
          f"(missing={len(getattr(missing, 'missing_keys', missing))}, "
          f"unexpected={len(getattr(unexpected, 'unexpected_keys', unexpected))})")


def load_model_weights_excluding_recon(model: TransMotion3DP, ckpt_path: str, device="cpu"):
    """
    再構成モデル以外(それ以外)の重みを読み込む。
    stgcn / coord_decoder は上書きしない。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = _resolve_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(sd)}")

    remapped = {}
    for k, v in sd.items():
        k = _strip_module_prefix(k)
        remapped[k] = v

    cur = model.state_dict()
    matched = {
        k: v for k, v in remapped.items()
        if not (k.startswith("stgcn.") or k.startswith("coord_decoder."))
        and k in cur and cur[k].shape == v.shape
    }
    cur.update(matched)
    missing, unexpected = model.load_state_dict(cur, strict=False)
    print(f"[model] loaded {len(matched)} tensors "
          f"(missing={len(getattr(missing, 'missing_keys', missing))}, "
          f"unexpected={len(getattr(unexpected, 'unexpected_keys', unexpected))})")

def load_and_freeze_backbone_for_transmotion(model: TransMotion3DP, ckpt_path: str, device='cpu'):
    """
    ckpt の 'encoder_state_dict'（または 'state_dict'）にある
      - 'encoder.*'           → model.stgcn.*
    を読み込んで、両モジュールを凍結（勾配停止＋BNをeval固定）する。
    """
    # Snapshot current ST-GCN weights for diff
    before = {k: v.detach().cpu().clone() for k, v in model.stgcn.state_dict().items()}

    ckpt = torch.load(ckpt_path, map_location=device)
    # Resolve checkpoint payload to a state_dict-like mapping
    if isinstance(ckpt, dict):
        sd = ckpt.get('encoder_state_dict',
             ckpt.get('model_state_dict',
             ckpt.get('state_dict', ckpt)))
    else:
        sd = ckpt

    # DataParallel対策: 'module.' を剥がす
    def strip_module(k): 
        return k[7:] if k.startswith('module.') else k

    remapped = {}
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(sd)}")
    for k, v in sd.items():
        k = strip_module(k)
        # if k.endswith((".running_mean", ".running_var", ".num_batches_tracked")):
        #     continue
        if k.startswith('stgcn.'):
            remapped[k] = v
            print(f"loaded {k} (direct)")
        elif k.startswith('encoder.'):
            new_k = 'stgcn.' + k[len('encoder.'):]
            remapped[new_k] = v
            print(f"loaded {k} -> {new_k}")
        elif k.startswith('coord_to_feature.'):
            # ignore non-stgcn weights
            continue


    # 既存 state_dict に上書き（形状が一致するキーのみ）
    cur = model.state_dict()
    matched = {k: v for k, v in remapped.items() if k in cur and cur[k].shape == v.shape}
    cur.update(matched)
    missing, unexpected = model.load_state_dict(cur, strict=False)

    if len(matched) == 0:
        print("[backbone] WARNING: no stgcn weights matched. Check checkpoint keys.")
    else:
        # Report max diff after load
        after = {k: v.detach().cpu() for k, v in model.stgcn.state_dict().items()}
        diffs = []
        for k in matched.keys():
            k_sub = k[len("stgcn."):] if k.startswith("stgcn.") else k
            if k_sub in after and k_sub in before:
                diffs.append((after[k_sub] - before[k_sub]).abs().max().item())
        if diffs:
            print(f"[backbone] max |Δ| in stgcn after load: {max(diffs):.6f}")
        else:
            sample_keys = list(matched.keys())[:5]
            print("[backbone] WARNING: diff check skipped (no overlapping keys).")
            print(f"[backbone] matched sample keys: {sample_keys}")
    print(f"[backbone] loaded {len(matched)} tensors "
          f"(missing={len(getattr(missing, 'missing_keys', missing))}, "
          f"unexpected={len(getattr(unexpected, 'unexpected_keys', unexpected))})")

    # ---- ST-GCNは凍結（BNも含めて固定） ----
    for p in model.stgcn.parameters():
        p.requires_grad = False

    for m in model.stgcn.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            # m.track_running_stats = True
            m.track_running_stats = False
            if m.affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    print("[backbone] stgcn frozen (including BN).")

