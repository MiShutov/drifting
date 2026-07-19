import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict


def compute_drift(
    x: torch.Tensor,                      # [N_features, N_gen, D]
    y_pos: torch.Tensor,                  # [N_features, N_pos, D]
    y_neg: Optional[torch.Tensor] = None, # [N_features, N_neg, D] or None
    T_list: List[float] = [0.02, 0.05, 0.2],
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Вычисляет суммарное дрейфующее поле V для всех температур.
    Теперь без цикла!
    """
    N_features, N_x, D = x.shape
    N_pos = y_pos.shape[1]
    N_temp = len(T_list)
    
    x = x.float()
    y_pos = y_pos.float()
    y_neg = y_neg.float() if y_neg is not None else y_neg

    if y_neg is None:
        y_neg = x
        mask_self = True
    else:
        mask_self = False
    N_neg = y_neg.shape[1]
    
    # --- Веса ---
    if weight_gen is None:
        weight_gen = torch.ones(N_features, N_x, device=x.device, dtype=x.dtype)
    if weight_pos is None:
        weight_pos = torch.ones(N_features, N_pos, device=x.device, dtype=x.dtype)
    if weight_neg is None:
        weight_neg = torch.ones(N_features, N_neg, device=x.device, dtype=x.dtype)
    
    # --- Расстояния ---
    dist_pos = torch.cdist(x, y_pos)  # [N_features, N_x, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N_features, N_x, N_neg]
    
    if mask_self:
        eye = torch.eye(N_x, device=x.device, dtype=torch.bool).unsqueeze(0) # [1, N_x, N_x]
        dist_neg = dist_neg.masked_fill(eye, 1e6)
    
    # --- Веса для scale ---
    w_pos = weight_gen.unsqueeze(-1) * weight_pos.unsqueeze(1)  # [N_features, N_x, N_pos]
    w_neg = weight_gen.unsqueeze(-1) * weight_neg.unsqueeze(1)  # [N_features, N_x, N_neg]
    
    # --- Масштаб (один для всех температур) ---
    weighted_sum = (w_pos * dist_pos).sum(dim=[1, 2]) + (w_neg * dist_neg).sum(dim=[1, 2])  # [N_features]
    total_weight = w_pos.sum(dim=[1, 2]) + w_neg.sum(dim=[1, 2])  # [N_features]
    scale = weighted_sum / (total_weight + 1e-8)  # [N_features]
    scale = torch.clamp(scale, min=1e-6)
    
    # --- Нормализация ---
    scale_inputs = scale / torch.sqrt(torch.tensor(D, dtype=x.dtype, device=x.device))
    scale_inputs = torch.clamp(scale_inputs, min=1e-3).view(N_features, 1, 1)
    
    x_scaled = x / scale_inputs
    y_pos_scaled = y_pos / scale_inputs
    y_neg_scaled = y_neg / scale_inputs
    
    dist_pos_norm = dist_pos / scale.view(N_features, 1, 1)
    dist_neg_norm = dist_neg / scale.view(N_features, 1, 1)
    
    # --- ВЕКТОРИЗАЦИЯ ПО ТЕМПЕРАТУРАМ! ---
    # Добавляем измерение для температур
    T_tensor = torch.tensor(T_list, device=x.device, dtype=x.dtype).view(N_temp, 1, 1, 1)  # [N_temp, 1, 1, 1]
    
    # Расширяем размерности для broadcast
    dist_pos_norm_exp = dist_pos_norm.unsqueeze(0)  # [1, N_features, N_x, N_pos]
    dist_neg_norm_exp = dist_neg_norm.unsqueeze(0)  # [1, N_features, N_x, N_neg]
    
    # Вычисляем логиты для всех температур сразу
    logit_pos = -dist_pos_norm_exp / T_tensor  # [N_temp, N_features, N_x, N_pos]
    logit_neg = -dist_neg_norm_exp / T_tensor  # [N_temp, N_features, N_x, N_neg]
    logit = torch.cat([logit_pos, logit_neg], dim=3)  # [N_temp, N_features, N_x, N_pos+N_neg]
    
    # Softmax по обеим осям
    A_row = F.softmax(logit, dim=-1)  # [N_temp, N_features, N_x, N_total]
    A_col = F.softmax(logit, dim=-2)  # [N_temp, N_features, N_x, N_total]
    A = torch.sqrt(A_row * A_col + 1e-8)  # [N_temp, N_features, N_x, N_total]
    
    # Разделяем
    A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=3)  # [N_temp, N_features, N_x, N_pos/neg]
    
    # --- Коэффициенты ---
    sum_pos = A_pos.sum(dim=3, keepdim=True)  # [N_temp, N_features, N_x, 1]
    sum_neg = A_neg.sum(dim=3, keepdim=True)  # [N_temp, N_features, N_x, 1]
    
    r_coeff_neg = -A_neg * sum_pos  # [N_temp, N_features, N_x, N_neg]
    r_coeff_pos = A_pos * sum_neg   # [N_temp, N_features, N_x, N_pos]
    R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=3)  # [N_temp, N_features, N_x, N_total]
    
    # --- Веса target ---
    weights_target = torch.cat([weight_neg, weight_pos], dim=1)  # [N_features, N_neg+N_pos]
    weights_target_exp = weights_target.unsqueeze(0).unsqueeze(2)  # [1, N_features, 1, N_total]
    R_coeff = R_coeff * weights_target_exp  # [N_temp, N_features, N_x, N_total]
    
    targets_scaled = torch.cat([y_neg_scaled, y_pos_scaled], dim=1)  # [N_features, N_total, D]
    targets_scaled_exp = targets_scaled.unsqueeze(0)  # [1, N_features, N_total, D]
    
    # --- Матричное умножение для всех температур ---
    # R_coeff: [N_temp, N_features, N_x, N_total]
    # targets_scaled_exp: [1, N_features, N_total, D]
    # Результат: [N_temp, N_features, N_x, D]
    total_force_R = torch.matmul(R_coeff, targets_scaled_exp)  # используем matmul для 4D
    
    total_coeffs = R_coeff.sum(dim=3, keepdim=True)  # [N_temp, N_features, N_x, 1]
    total_force_R = total_force_R - total_coeffs * x_scaled.unsqueeze(0)  # [N_temp, N_features, N_x, D]
    
    # --- Нормализация силы для каждой температуры ---
    f_norm_val = (total_force_R ** 2).mean(dim=[2, 3], keepdim=True)  # [N_temp, N_features, 1, 1]
    force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))  # [N_temp, N_features, 1, 1]
    V_temp = total_force_R / force_scale  # [N_temp, N_features, N_x, D]
    
    # --- Усредняем по температурам ---
    V = V_temp.mean(dim=0)  # [N_features, N_x, D]
    
    return V


def drifting_loss(
    x: torch.Tensor,                      # [N_features, N_gen, D]
    y_pos: torch.Tensor,                  # [N_features, N_pos, D]
    y_neg: Optional[torch.Tensor] = None, # [N_features, N_neg, D] or None
    T_list: List[float] = [0.02, 0.05, 0.2],
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Drifting loss
    """
    with torch.no_grad():
        V = compute_drift(
            x, 
            y_pos, 
            y_neg,
            T_list,
            weight_gen, 
            weight_pos,
            weight_neg
        )
        target = x + V
    return F.mse_loss(x, target.detach())


def drifting_loss_multifeatures(
    x_features: List[torch.Tensor],                                 # N_feature_types x [N_features, N_x, D_feature_type]
    y_pos_features: List[torch.Tensor],                             # N_feature_types x [N_features, N_pos, D_feature_type]
    y_neg_features: Optional[List[Optional[torch.Tensor]]] = None,  # N_feature_types x [N_features, y_neg, D_feature_type]
    T_list: List[float] = [0.02, 0.05, 0.2],
    weight_gen_list: Optional[List[Optional[torch.Tensor]]] = None,
    weight_pos_list: Optional[List[Optional[torch.Tensor]]] = None,
    weight_neg_list: Optional[List[Optional[torch.Tensor]]] = None,
) -> torch.Tensor:
    total_loss = 0.0
    n = len(x_features)
    if y_neg_features is None:
        y_neg_features = [None] * n
    if weight_gen_list is None:
        weight_gen_list = [None] * n
    if weight_pos_list is None:
        weight_pos_list = [None] * n
    if weight_neg_list is None:
        weight_neg_list = [None] * n

    if isinstance(x_features, dict):
        x_features = list(x_features.values())
    if isinstance(y_pos_features, dict):
        y_pos_features = list(y_pos_features.values())
    if isinstance(y_neg_features, dict):
        y_neg_features = list(y_neg_features.values())


    for i in range(n):
        loss_i = drifting_loss(
            x_features[i], 
            y_pos_features[i], 
            y_neg_features[i],
            T_list, 
            weight_gen_list[i], 
            weight_pos_list[i], 
            weight_neg_list[i]
        )
        total_loss = total_loss + loss_i
    return total_loss

