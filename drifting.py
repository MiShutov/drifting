import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict


# def compute_drift_single_temp(
#     x: torch.Tensor,                # [N_features, N_gen, D]
#     y_pos: torch.Tensor,            # [N_features, N_pos, D]
#     y_neg: Optional[torch.Tensor],  # [N_features, N_neg, D] or None
#     T: float,
#     weight_gen: Optional[torch.Tensor] = None,  # [N_features, N_x] or None
#     weight_pos: Optional[torch.Tensor] = None,  # [N_features, N_pos] or None
#     weight_neg: Optional[torch.Tensor] = None,  # [N_features, N_neg] or None
# ) -> Tuple[torch.Tensor, Dict]:
#     N_features, N_x, D = x.shape
#     N_pos = y_pos.shape[1]
#     if y_neg is None:
#         y_neg = x
#         mask_self = True
#     else:
#         mask_self = False
#     N_neg = y_neg.shape[1]


#     if weight_gen is None:
#         weight_gen = torch.ones(N_features, N_x, device=x.device, dtype=x.dtype)
#     if weight_pos is None:
#         weight_pos = torch.ones(N_features, N_pos, device=x.device, dtype=x.dtype)
#     if weight_neg is None:
#         weight_neg = torch.ones(N_features, N_neg, device=x.device, dtype=x.dtype)


#     dist_pos = torch.cdist(x, y_pos)  # [N_features, N_x, N_pos]
#     dist_neg = torch.cdist(x, y_neg)  # [N_features, N_x, N_neg]

#     if mask_self:
#         eye = torch.eye(N_x, device=x.device, dtype=torch.bool)
#         dist_neg = dist_neg.masked_fill(eye, 1e6)

#     w_pos = weight_gen.unsqueeze(-1) * weight_pos.unsqueeze(1)  # [N_features, N_x, N_pos]
#     w_neg = weight_gen.unsqueeze(-1) * weight_neg.unsqueeze(1)  # [N_features, N_x, N_neg]

#     weighted_sum = (w_pos * dist_pos).sum(dim=[1, 2]) + (w_neg * dist_neg).sum(dim=[1, 2])  # [N_features]
#     total_weight = w_pos.sum(dim=[1, 2]) + w_neg.sum(dim=[1, 2])  # [N_features]
#     scale = weighted_sum / (total_weight + 1e-8)  # [N_features]
#     scale = torch.clamp(scale, min=1e-6) # [N_features]

#     scale_inputs = scale / torch.sqrt(torch.tensor(D, dtype=x.dtype, device=x.device))
#     scale_inputs = torch.clamp(scale_inputs, min=1e-3).view(N_features, 1, 1)  # [N_features, 1, 1]

#     x_scaled = x / scale_inputs
#     y_pos_scaled = y_pos / scale_inputs
#     y_neg_scaled = y_neg / scale_inputs

#     dist_pos_norm = dist_pos / scale.view(N_features, 1, 1)
#     dist_neg_norm = dist_neg / scale.view(N_features, 1, 1)

#     logit_pos = -dist_pos_norm / T
#     logit_neg = -dist_neg_norm / T
#     logit = torch.cat([logit_pos, logit_neg], dim=2)  # [N_features, N_x, N_pos+N_neg]

#     A_row = F.softmax(logit, dim=-1)  # [N_features, N_x, N_total]
#     A_col = F.softmax(logit, dim=-2)  # [N_features, N_x, N_total]
#     A = torch.sqrt(A_row * A_col + 1e-8)

#     A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=2)  # [N_features, N_x, N_pos], [N_features, N_x, N_neg]

#     sum_pos = A_pos.sum(dim=2, keepdim=True)  # [N_features, N_x, 1]
#     sum_neg = A_neg.sum(dim=2, keepdim=True)  # [N_features, N_x, 1]

#     r_coeff_neg = -A_neg * sum_pos  # [N_features, N_x, N_neg]
#     r_coeff_pos = A_pos * sum_neg   # [N_features, N_x, N_pos]
#     R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)  # [N_features, N_x, N_pos+N_neg]

#     weights_target = torch.cat([weight_neg, weight_pos], dim=1)  # [N_features, N_neg+N_pos]
#     R_coeff = R_coeff * weights_target.unsqueeze(1)  # [N_features, N_x, N_neg+N_pos]
    
#     targets_scaled = torch.cat([y_neg_scaled, y_pos_scaled], dim=1)  # [N_features, N_neg+N_pos, D]

#     total_force_R = torch.bmm(R_coeff, targets_scaled)  # [N_features, N_x, D]
    
#     total_coeffs = R_coeff.sum(dim=2, keepdim=True)  # [N_features, N_x, 1]
#     total_force_R = total_force_R - total_coeffs * x_scaled # [N_features, N_x, D]

#     f_norm_val = (total_force_R ** 2).mean(dim=[1, 2])  # [N_features]
#     force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8)).view(N_features, 1, 1)
#     V = total_force_R / force_scale  # [N_features, N_x, D]
#     return V


# def compute_drift(
#     x: torch.Tensor,                      # [N_features, N_gen, D]
#     y_pos: torch.Tensor,                  # [N_features, N_pos, D]
#     y_neg: Optional[torch.Tensor] = None, # [N_features, N_neg, D] or None
#     T_list: List[float] = [0.02, 0.05, 0.2],
#     weight_gen: Optional[torch.Tensor] = None,
#     weight_pos: Optional[torch.Tensor] = None,
#     weight_neg: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, Dict]:
#     V_total = torch.zeros_like(x)
#     for T in T_list:
#         V = compute_drift_single_temp(
#             x, y_pos, y_neg, T,
#             weight_gen, weight_pos, weight_neg
#         )
#         V_total += V
#     V_avg = V_total / len(T_list)
#     return V_avg


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
            x, y_pos, y_neg, T_list,
            weight_gen, weight_pos, weight_neg
        )
        target = x + V
    return F.mse_loss(x, target.detach())


def drifting_loss_multifeatures(
    x_list: List[torch.Tensor],                                 # N_feature_types x [N_features, N_x, D_feature_type]
    y_pos_list: List[torch.Tensor],                             # N_feature_types x [N_features, N_pos, D_feature_type]
    y_neg_list: Optional[List[Optional[torch.Tensor]]] = None,  # N_feature_types x [N_features, y_neg, D_feature_type]
    T_list: List[float] = [0.02, 0.05, 0.2],
    weight_gen_list: Optional[List[Optional[torch.Tensor]]] = None,
    weight_pos_list: Optional[List[Optional[torch.Tensor]]] = None,
    weight_neg_list: Optional[List[Optional[torch.Tensor]]] = None,
) -> torch.Tensor:
    total_loss = 0.0
    n = len(x_list)
    if y_neg_list is None:
        y_neg_list = [None] * n
    if weight_gen_list is None:
        weight_gen_list = [None] * n
    if weight_pos_list is None:
        weight_pos_list = [None] * n
    if weight_neg_list is None:
        weight_neg_list = [None] * n

    for i in range(n):
        loss_i = drifting_loss(
            x_list[i], y_pos_list[i], y_neg_list[i],
            T_list, weight_gen_list[i], weight_pos_list[i], weight_neg_list[i]
        )
        total_loss = total_loss + loss_i
    return total_loss


# import torch
# import torch.nn.functional as F


# def compute_drift_single_temp(x, y_pos, y_neg=None, T=0.05):
#     """
#     Реализация из Algorithm 2 статьи.
#     x: [B, D] - сгенерированные образцы
#     y_pos: [B, D] - положительные (реальные) образцы
#     y_neg: [B, D] - отрицательные (другие сгенерированные), опционально
#     """
#     B, D = x.shape
    
#     # Если отрицательные не заданы, используем x (как в статье)
#     if y_neg is None:
#         y_neg = x.clone()
    
#     # Вычисляем расстояния
#     dist_pos = torch.cdist(x, y_pos)  # [B, B]
#     dist_neg = torch.cdist(x, y_neg)  # [B, B]
    
#     # Маскируем диагональ для отрицательных (чтобы не сравнивать с собой)
#     eye = torch.eye(B, device=x.device)
#     dist_neg = dist_neg + eye * 1e6
    
#     # Логиты
#     logit_pos = -dist_pos / T
#     logit_neg = -dist_neg / T
    
#     # Объединяем для softmax
#     logit = torch.cat([logit_pos, logit_neg], dim=1)  # [B, 2B]
    
#     # Softmax по обеим осям (как в Algorithm 2)
#     A_row = F.softmax(logit, dim=-1)
#     A_col = F.softmax(logit, dim=-2)
#     A = torch.sqrt(A_row * A_col + 1e-8)
    
#     # Разделяем обратно
#     A_pos, A_neg = torch.split(A, B, dim=1)  # [B, B], [B, B]
    
#     # Вычисляем веса как в статье
#     W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [B, B]
#     W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [B, B]
    
#     # Дрейфующее поле
#     V = (W_pos @ y_pos) - (W_neg @ y_neg)  # [B, D]
    
#     return V


# def compute_drift(x, y_pos, y_neg=None, T_list=[0.02, 0.05, 0.2]):
#     V_total = torch.zeros_like(x)
    
#     if not isinstance(T_list, list):
#         T_list = [T_list]

#     for T in T_list:
#         V = compute_drift_single_temp(x, y_pos, y_neg, T)
#         V_total += V
    
#     V_avg = V_total / len(T_list)
#     return V_avg


# def drifting_loss(
#         x: torch.Tensor, 
#         y_pos: torch.Tensor,
#         y_neg: torch.Tensor = None, 
#         compute_drift=compute_drift,
#     ):
#     """Drifting loss: MSE(gen, stopgrad(gen + V))."""

#     x = x.float()
#     y_pos = y_pos.float()
#     y_neg = y_neg.float() if y_neg is not None else None

#     with torch.no_grad():
#         V = compute_drift(x, y_pos, y_neg)
#         target = (x + V).detach()
#     return F.mse_loss(x, target)


# def drifting_loss_multifeatures(
#         x: list, 
#         y_pos: list,
#         y_neg: list = None, 
#         compute_drift=compute_drift,
#     ):
#     """Drifting loss: MSE(gen, stopgrad(gen + V))."""
#     total_loss = 0
#     n_steps = len(x)
#     for step in range(n_steps):
#         total_loss += drifting_loss(
#             x[step],
#             y_pos[step],
#             y_neg[step] if y_neg is not None else None,
#             compute_drift=compute_drift,
#         )

#     return total_loss




# def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """
#     Вычисляет попарные евклидовы расстояния между x и y.
#     x: [N, D], y: [M, D] -> [N, M]
#     """
#     x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [N, 1]
#     y_norm = (y ** 2).sum(dim=1, keepdim=True)  # [M, 1]
#     dot = torch.mm(x, y.t())                     # [N, M]
#     dist_sq = x_norm + y_norm.t() - 2 * dot
#     dist_sq = torch.clamp(dist_sq, min=eps)
#     return torch.sqrt(dist_sq)

# def compute_drift_single_temp(
#     x: torch.Tensor,                # [N_gen, D]
#     y_pos: torch.Tensor,            # [N_pos, D]
#     y_neg: Optional[torch.Tensor],  # [N_neg, D] or None
#     T: float,
#     weight_gen: Optional[torch.Tensor] = None,  # [N_gen] or None
#     weight_pos: Optional[torch.Tensor] = None,  # [N_pos] or None
#     weight_neg: Optional[torch.Tensor] = None,  # [N_neg] or None
# ) -> Tuple[torch.Tensor, Dict]:
#     """
#     Вычисляет дрейфующее поле V для одной температуры T.
#     Возвращает V (нормализованное) и словарь информации.
#     """
#     N_gen, D = x.shape
#     N_pos = y_pos.shape[0]
#     if y_neg is None:
#         y_neg = x
#         # маскируем диагональ, если отрицательные = сами сгенерированные
#         mask_self = True
#     else:
#         mask_self = False
#     N_neg = y_neg.shape[0]

#     # --- Веса ---
#     if weight_gen is None:
#         weight_gen = torch.ones(N_gen, device=x.device, dtype=x.dtype)
#     if weight_pos is None:
#         weight_pos = torch.ones(N_pos, device=x.device, dtype=x.dtype)
#     if weight_neg is None:
#         weight_neg = torch.ones(N_neg, device=x.device, dtype=x.dtype)

#     # --- Расстояния ---
#     dist_pos = cdist(x, y_pos)  # [N_gen, N_pos]
#     dist_neg = cdist(x, y_neg)  # [N_gen, N_neg]

#     # Маскировка самосравнения (если y_neg == x)
#     if mask_self:
#         eye = torch.eye(N_gen, device=x.device, dtype=torch.bool)
#         # dist_neg имеет размер [N_gen, N_gen], маскируем диагональ
#         dist_neg = dist_neg.masked_fill(eye, 1e6)

#     # --- Вычисление масштаба (scale) ---
#     # Среднее взвешенное расстояние по всем парам
#     # Учитываем веса как для gen, так и для target
#     # веса для target: конкатенация [weight_neg, weight_pos] для обеих дистанций
#     # Мы сделаем отдельно для pos и neg, потом объединим
#     # Для pos: вес пары = weight_gen[i] * weight_pos[j]
#     # Для neg: вес пары = weight_gen[i] * weight_neg[j]
#     # Общий знаменатель - сумма всех весов

#     # Считаем взвешенные суммы для pos и neg
#     w_pos = weight_gen[:, None] * weight_pos[None, :]  # [N_gen, N_pos]
#     w_neg = weight_gen[:, None] * weight_neg[None, :]  # [N_gen, N_neg]

#     weighted_sum_dist = (w_pos * dist_pos).sum() + (w_neg * dist_neg).sum()
#     total_weight = w_pos.sum() + w_neg.sum()
#     if total_weight > 0:
#         scale = weighted_sum_dist / total_weight
#     else:
#         scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)
#     scale = torch.clamp(scale, min=1e-6)

#     # Нормировочный коэффициент для признаков
#     scale_inputs = torch.clamp(scale / torch.sqrt(torch.tensor(D, dtype=x.dtype)), min=1e-3)

#     # Нормируем признаки и расстояния
#     x_scaled = x / scale_inputs
#     y_pos_scaled = y_pos / scale_inputs
#     y_neg_scaled = y_neg / scale_inputs

#     dist_normed_pos = dist_pos / scale
#     dist_normed_neg = dist_neg / scale

#     # --- Логиты и аффинити (softmax по обеим осям) ---
#     logit_pos = -dist_normed_pos / T
#     logit_neg = -dist_normed_neg / T

#     # Конкатенируем по оси target: [N_gen, N_pos + N_neg]
#     logit = torch.cat([logit_pos, logit_neg], dim=1)

#     # Softmax по строкам и столбцам
#     A_row = F.softmax(logit, dim=-1)       # [N_gen, N_total]
#     A_col = F.softmax(logit, dim=-2)       # [N_gen, N_total]
#     A = torch.sqrt(A_row * A_col + 1e-8)   # [N_gen, N_total]

#     # Разделяем обратно
#     A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)  # каждая [N_gen, N_pos/neg]

#     # --- Вычисление коэффициентов R_coeff (как в оригинале) ---
#     # sum по оси target для pos и neg
#     sum_pos = A_pos.sum(dim=1, keepdim=True)  # [N_gen, 1]
#     sum_neg = A_neg.sum(dim=1, keepdim=True)  # [N_gen, 1]

#     # Коэффициенты для отрицательных и положительных
#     r_coeff_neg = -A_neg * sum_pos           # [N_gen, N_neg]
#     r_coeff_pos = A_pos * sum_neg            # [N_gen, N_pos]

#     # Объединяем R_coeff (последовательность: сначала neg, потом pos)
#     R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=1)  # [N_gen, N_neg+N_pos]

#     # Умножаем на веса target (как в оригинале)
#     # веса для target: конкатенация weight_neg и weight_pos
#     weights_target = torch.cat([weight_neg, weight_pos], dim=0)  # [N_neg+N_pos]
#     R_coeff = R_coeff * weights_target[None, :]  # [N_gen, N_total]

#     # Целевые признаки (масштабированные) в том же порядке: сначала neg, потом pos
#     targets_scaled = torch.cat([y_neg_scaled, y_pos_scaled], dim=0)  # [N_neg+N_pos, D]

#     # Вычисляем total_force_R = einsum('ij,jk->ik', R_coeff, targets_scaled)
#     total_force_R = torch.mm(R_coeff, targets_scaled)  # [N_gen, D]

#     # Вычитаем total_coeffs * x_scaled (для численной стабильности, как в оригинале)
#     total_coeffs = R_coeff.sum(dim=1, keepdim=True)  # [N_gen, 1]
#     total_force_R = total_force_R - total_coeffs * x_scaled

#     # --- Нормировка силы по данной температуре ---
#     f_norm_val = (total_force_R ** 2).mean()  # скаляр
#     force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
#     V = total_force_R / force_scale  # [N_gen, D]

#     # Сохраняем информацию
#     info = {
#         'scale': scale.item(),
#         'loss_R': f_norm_val.item(),
#         'force_scale': force_scale.item(),
#     }

#     return V, info
