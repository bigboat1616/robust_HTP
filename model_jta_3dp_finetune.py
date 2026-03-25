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
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21, num_tokens=2, mask_ratio=0.0, mask_joints=None, device='cuda:0'):
        super(TransMotion3DP, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.joints_pose = 22
        self.obs_and_pred = 21
        self.device = device
        self.mask_ratio = mask_ratio
        self.mask_joints = mask_joints
        
        # Trajectory layers
        self.fc_in_traj = nn.Linear(2, nhid)
        self.fc_out_traj = nn.Linear(nhid, 2)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=21, device=device)
        self.id_encoder = LearnedIDEncoding(nhid, dropout, seq_len=21, device=device)

        
        # 3D pose layers
        self.fc_in_3dpose = nn.Linear(3, nhid)
        self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)


        # ST-GCN layers
        graph_cfg = {'layout': 'jta_3dp_row', 'strategy': 'distance', 'max_hop': 1, 'dilation': 1}
        self.stgcn = ST_GCN_18(in_channels=3, feature_dim=nhid, graph_cfg=graph_cfg, edge_importance_weighting=True, data_bn=True, layer_num=3)

        # Transformer layers
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
        # Padding handling
        i_idx = torch.cat([torch.arange(in_F), torch.full((out_F,), in_F-1)]).to(tgt.device)
        tgt = tgt[:,i_idx]        
        tgt = tgt.reshape(B,F,N,J,K)

        # Mask configuration
        mask_ratio = self.mask_ratio

        # Trajectory data processing
        tgt_traj = tgt[:,:,:,0,:2] 

        # 3D pose data processing
        tgt_3dpose = tgt[:,:,:,1:,:3]  
        # Mask exactly K joints (excluding joint 15)
        allowed_joints = torch.tensor(
            [j for j in range(self.joints_pose) if j != 15],
            device=self.device,
        )
        if self.mask_joints is None:
            num_mask = int(round(mask_ratio * allowed_joints.numel()))
        else:
            num_mask = int(self.mask_joints)
        num_mask = max(0, min(num_mask, allowed_joints.numel()))
        joints_3d_mask = torch.zeros((B, F, N, self.joints_pose), device=self.device, dtype=torch.bool)
        if num_mask > 0:
            scores = torch.rand((B, F, N, allowed_joints.numel()), device=self.device)
            _, idx = scores.topk(num_mask, dim=-1)
            selected = allowed_joints[idx]
            joints_3d_mask.scatter_(dim=-1, index=selected, value=True)
        mask_token = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=tgt_3dpose.dtype)
        tgt_3dpose = torch.where(joints_3d_mask.unsqueeze(-1), mask_token, tgt_3dpose)

        # Encoding
        tgt_traj = self.fc_in_traj(tgt_traj) 
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N) 

        BN = B * N
        # print("3dpose_row mean/std:", tgt_3dpose.mean().item(), tgt_3dpose.std().item())
        tgt_3dpose = tgt_3dpose[:,:9]  # (B, 9, N, 22, 3)
        tgt_3dpose = tgt_3dpose.permute(0, 2, 4, 1, 3).contiguous() 
        tgt_3dpose = tgt_3dpose.reshape(BN, 3, 9, 22) 
        # Forward through ST-GCN
        tgt_3dpose = self.stgcn(tgt_3dpose)  # (B*N, nhid, 9, 22)

        # print("stgcn mean/std:", tgt_3dpose.mean().item(), tgt_3dpose.std().item())
        # Match existing flow: reshape to (B, 198, N, nhid)
        tgt_3dpose = tgt_3dpose.reshape(B, N, self.nhid, 9, 22)
        tgt_3dpose = tgt_3dpose.permute(0, 3, 4, 1, 2).contiguous()  # (B, 198, N, nhid)
        tgt_3dpose = tgt_3dpose.reshape(B, 9 * 22, N, self.nhid)
        tgt_3dpose = self.pose3d_encoder(tgt_3dpose)

        # Padding mask handling
        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1) 
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1) 
  
        # Reshape data
        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid) 
        tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_pose, -1, self.nhid) 
        # Zero out 3D pose for ablation
        # tgt_3dpose = torch.zeros(in_F*self.joints_pose, B*N , self.nhid).to(self.device)
        # print("traj mean/std:", tgt_traj.mean().item(), tgt_traj.std().item())
        # print("3dpose mean/std:", tgt_3dpose.mean().item(), tgt_3dpose.std().item())
        # Concatenate
        tgt = torch.cat((tgt_traj, tgt_3dpose), 0) 
        # Transformer forward
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
    mask_ratio = config["MODEL"].get("mask_rate", 0.0)
    mask_joints = config["MODEL"].get("mask_joints", None)
    model = TransMotion3DP(tok_dim=seq_len,
        nhid=nhid,
        nhead=nhead,
        dim_feedfwd=dim_feedforward,
        nlayers_local=nlayers_local,
        nlayers_global=nlayers_global,
        output_scale=config["MODEL"]["output_scale"],
        obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
        num_tokens=token_num,
        mask_ratio=mask_ratio,
        mask_joints=mask_joints,
        device=config["DEVICE"]
    ).to(config["DEVICE"]).float()

    # Load and freeze the pretrained backbone only during training.
    ckpt_path = config["MODEL"].get("backbone_ckpt")
    if model.training and ckpt_path:
        load_and_freeze_backbone_for_transmotion(model, ckpt_path, device=config["DEVICE"])
    elif ckpt_path:
        pass
        # print("[backbone] skipped loading because model is not in train() mode.")
    else:
        pass
        # print("[backbone] no backbone_ckpt provided; skipping load.")
    return model 

def load_and_freeze_backbone_for_transmotion(model: TransMotion3DP, ckpt_path: str, device='cpu'):
    """
    In ckpt['encoder_state_dict'] (or 'state_dict'),
      - 'encoder.*'           → model.stgcn.*
    load weights and freeze both modules (no grads + BN eval).
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

    # DataParallel: strip 'module.' prefix
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
            # print(f"loaded {k} (direct)")
        elif k.startswith('encoder.'):
            new_k = 'stgcn.' + k[len('encoder.'):]
            remapped[new_k] = v
            # print(f"loaded {k} -> {new_k}")
        elif k.startswith('coord_to_feature.'):
            # ignore non-stgcn weights
            continue


    # Update current state_dict (keys with matching shapes only)
    cur = model.state_dict()
    matched = {k: v for k, v in remapped.items() if k in cur and cur[k].shape == v.shape}
    cur.update(matched)
    missing, unexpected = model.load_state_dict(cur, strict=False)

    if len(matched) == 0:
        pass
        # print("[backbone] WARNING: no stgcn weights matched. Check checkpoint keys.")
    else:
        # Report max diff after load
        after = {k: v.detach().cpu() for k, v in model.stgcn.state_dict().items()}
        diffs = []
        for k in matched.keys():
            k_sub = k[len("stgcn."):] if k.startswith("stgcn.") else k
            if k_sub in after and k_sub in before:
                diffs.append((after[k_sub] - before[k_sub]).abs().max().item())
        if diffs:
            pass
            # print(f"[backbone] max |Δ| in stgcn after load: {max(diffs):.6f}")
        else:
            pass
            # sample_keys = list(matched.keys())[:5]
            # print("[backbone] WARNING: diff check skipped (no overlapping keys).")
            # print(f"[backbone] matched sample keys: {sample_keys}")
    # print(f"[backbone] loaded {len(matched)} tensors "
    #       f"(missing={len(getattr(missing, 'missing_keys', missing))}, "
    #       f"unexpected={len(getattr(unexpected, 'unexpected_keys', unexpected))})")

    # ---- Freeze ST-GCN (including BN) ----
    model.stgcn.eval()
    for p in model.stgcn.parameters():
        p.requires_grad = False

    for m in model.stgcn.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.track_running_stats = True
            if m.affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    # print("[backbone] stgcn frozen (including BN).")

