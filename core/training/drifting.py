import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict


@torch.compile
def compute_drift(
    x: torch.Tensor,                      # [N_features, N_gen, D]
    y_pos: torch.Tensor,                  # [N_features, N_pos, D]
    y_neg: Optional[torch.Tensor] = None, # [N_features, N_neg, D] or None
    T_list: List[float] = [0.02, 0.05, 0.2],
    force_fp32=False
) -> torch.Tensor:
    log_dict = {}

    N_features, N_x, D = x.shape
    N_pos = y_pos.shape[1]
    N_temp = len(T_list)

    x_dtype = x.dtype

    if force_fp32:
        x = x.float()
        y_pos = y_pos.float()
        y_neg = y_neg.float() if y_neg is not None else y_neg

    if y_neg is None:
        y_neg = x
        mask_self = True
    else:
        mask_self = False
    N_neg = y_neg.shape[1]
    
    # --- Dist computation ---
    dist_pos = torch.cdist(x, y_pos)  # [N_features, N_x, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N_features, N_x, N_neg]

    # --- Scaling ---
    scale = (dist_pos.mean(dim=[1, 2]) + dist_neg.mean(dim=[1, 2])) / 2  # [N_features]
    dist_pos = dist_pos / scale.view(N_features, 1, 1)
    dist_neg = dist_neg / scale.view(N_features, 1, 1)

    # --- Different temperatures logits ---
    T_tensor = torch.tensor(T_list, device=x.device, dtype=x.dtype).view(N_temp, 1, 1, 1)  # [N_temp, 1, 1, 1]
    logit_pos = -dist_pos.unsqueeze(0) / T_tensor  # [N_temp, N_features, N_x, N_pos]
    logit_neg = -dist_neg.unsqueeze(0) / T_tensor  # [N_temp, N_features, N_x, N_neg]
    
    if mask_self:
        eye = torch.eye(N_x, device=x.device, dtype=torch.bool).unsqueeze(0) # [1, N_x, N_x]
        logit_neg = logit_neg.masked_fill(eye, -100)

    A_pos = F.softmax(logit_pos, dim=-1)  # [N_temp, N_features, N_x, N_pos]
    A_neg = F.softmax(logit_neg, dim=-1)  # [N_temp, N_features, N_x, N_neg]

    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)  # [N_temp, N_features, N_x, N_pos]
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)  # [N_temp, N_features, N_x, N_neg]

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    V = drift_pos - drift_neg

    f_norm_val = (V ** 2).mean(dim=[2, 3], keepdim=True)  # [N_temp, N_features, 1, 1]
    force_scale = torch.sqrt(f_norm_val)  # [N_temp, N_features, 1, 1]
    log_dict["V_abs"] = force_scale.mean().item()

    V = V.mean(dim=0)  # [N_features, N_x, D]
    return V.to(x_dtype), log_dict


def compute_drift_(
    x: torch.Tensor,                      # [N_features, N_gen, D]
    y_pos: torch.Tensor,                  # [N_features, N_pos, D]
    y_neg: Optional[torch.Tensor] = None, # [N_features, N_neg, D] or None
    T_list: List[float] = [0.02, 0.05, 0.2],
    force_fp32=False
) -> torch.Tensor:
    log_dict = {}

    N_features, N_x, D = x.shape
    N_pos = y_pos.shape[1]
    N_temp = len(T_list)

    x_dtype = x.dtype

    if force_fp32:
        x = x.float()
        y_pos = y_pos.float()
        y_neg = y_neg.float() if y_neg is not None else y_neg

    if y_neg is None:
        y_neg = x
        mask_self = True
    else:
        mask_self = False
    N_neg = y_neg.shape[1]
    
    # --- Dist computation ---
    dist_pos = torch.cdist(x, y_pos)  # [N_features, N_x, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N_features, N_x, N_neg]

    # --- Scaling ---
    scale = (dist_pos.mean(dim=[1, 2]) + dist_neg.mean(dim=[1, 2])) / 2  # [N_features]
    scale_inputs = scale / torch.sqrt(torch.tensor(D, dtype=x.dtype, device=x.device))
    scale_inputs = torch.clamp(scale_inputs, min=1e-3)

    x_scaled = x / scale_inputs.view(N_features, 1, 1)
    y_pos_scaled = y_pos / scale_inputs.view(N_features, 1, 1)
    y_neg_scaled = y_neg / scale_inputs.view(N_features, 1, 1)
    
    dist_pos = dist_pos / scale.view(N_features, 1, 1)
    dist_neg = dist_neg / scale.view(N_features, 1, 1)

    # --- Different temperatures logits ---
    T_tensor = torch.tensor(T_list, device=x.device, dtype=x.dtype).view(N_temp, 1, 1, 1)  # [N_temp, 1, 1, 1]
    logit_pos = -dist_pos.unsqueeze(0) / T_tensor  # [N_temp, N_features, N_x, N_pos]
    logit_neg = -dist_neg.unsqueeze(0) / T_tensor  # [N_temp, N_features, N_x, N_neg]

    if mask_self:
        eye = torch.eye(N_x, device=x.device, dtype=torch.bool).unsqueeze(0) # [1, N_x, N_x]
        logit_neg = logit_neg.masked_fill(eye, -100)
    logit = torch.cat([logit_pos, logit_neg], dim=3)  # [N_temp, N_features, N_x, N_pos+N_neg]


    # 2-axes softmax 
    A_row = F.softmax(logit, dim=-1)  # [N_temp, N_features, N_x, N_total]
    A_col = F.softmax(logit, dim=-2)  # [N_temp, N_features, N_x, N_total]
    A = torch.sqrt(A_row * A_col + 1e-8)  # [N_temp, N_features, N_x, N_total]
    A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=3)  # [N_temp, N_features, N_x, N_pos/neg]

    r_coeff_neg = -A_neg * A_pos.sum(dim=-1, keepdim=True)  # [N_temp, N_features, N_x, N_neg]
    r_coeff_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)   # [N_temp, N_features, N_x, N_pos]
    R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=3)  # [N_temp, N_features, N_x, N_total]
    
    total_force = torch.matmul(
        R_coeff, 
        torch.cat([y_neg_scaled, y_pos_scaled], dim=1)
    )
    total_coeffs = R_coeff.sum(dim=-1, keepdim=True)  # [N_temp, N_features, N_x, 1]
    total_force = total_force - total_coeffs * x_scaled.unsqueeze(0)  # [N_temp, N_features, N_x, D]    

    # # --- Temperature normalization ---
    f_norm_val = (total_force ** 2).mean(dim=[2, 3], keepdim=True)  # [N_temp, N_features, 1, 1]
    f_norm_val = torch.clamp(f_norm_val, min=1e-8)
    force_scale = torch.sqrt(f_norm_val)  # [N_temp, N_features, 1, 1]
    # V_temp = total_force / force_scale  # [N_temp, N_features, N_x, D]
    V_temp = total_force # [N_temp, N_features, N_x, D]

    log_dict["V_abs"] = force_scale.mean().item()

    V = V_temp.mean(dim=0)  # [N_features, N_x, D]
    return V.to(x_dtype), log_dict


def drifting_loss(
    x: torch.Tensor,                      # [N_features, N_gen, D]
    y_pos: torch.Tensor,                  # [N_features, N_pos, D]
    y_neg: Optional[torch.Tensor] = None, # [N_features, N_neg, D] or None
    T_list: List[float] = [0.02, 0.05, 0.2],
    force_fp32=False,
) -> torch.Tensor:
    """
    Drifting loss
    """
    with torch.no_grad():
        V, log_dict = compute_drift(
            x, 
            y_pos, 
            y_neg,
            T_list,
            force_fp32=force_fp32
        )
        target = x + V
    return F.mse_loss(x, target.detach()), log_dict


def drifting_loss_multifeatures(
    x_features: List[torch.Tensor],                                 # N_feature_types x [N_features, N_x, D_feature_type]
    y_pos_features: List[torch.Tensor],                             # N_feature_types x [N_features, N_pos, D_feature_type]
    y_neg_features: Optional[List[Optional[torch.Tensor]]] = None,  # N_feature_types x [N_features, y_neg, D_feature_type]
    T_list: List[float] = [0.02, 0.05, 0.2],
    force_fp32=False
) -> torch.Tensor:
    total_loss = 0.0
    n = len(x_features)
    if y_neg_features is None:
        y_neg_features = [None] * n

    if isinstance(x_features, dict):
        x_features = list(x_features.values())
    if isinstance(y_pos_features, dict):
        y_pos_features = list(y_pos_features.values())
    if isinstance(y_neg_features, dict):
        y_neg_features = list(y_neg_features.values())


    dist = 0
    logs = {}
    for i in range(n):
        loss_i, log_dict = drifting_loss(
            x_features[i], 
            y_pos_features[i], 
            y_neg_features[i],
            T_list,
            force_fp32=force_fp32
        )
        total_loss = total_loss + loss_i
        logs[i] = log_dict

    return total_loss, logs
