import torch
import torch.nn.functional as F


def drifting_loss(gen: torch.Tensor, pos: torch.Tensor, compute_drift):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


def compute_drift(
    gen: torch.Tensor, 
    pos: torch.Tensor,
    temp_list: list = [0.02, 0.05, 0.2],
    weight_gen: torch.Tensor = None,
    weight_pos: torch.Tensor = None,
    neg: torch.Tensor = None,
    weight_neg: torch.Tensor = None,
):
    """
    Compute drift field V with attention-based kernel.
    
    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp_list: List of temperatures for softmax kernel
        weight_gen: Weights for gen samples [G] (optional)
        weight_pos: Weights for pos samples [P] (optional)
        neg: Negative samples [N, D] (optional)
        weight_neg: Weights for neg samples [N] (optional)
    
    Returns:
        V: Drift vectors [G, D]
    """
    G, D = gen.shape
    
    # Handle optional parameters
    if neg is None:
        neg = torch.zeros(0, D, device=gen.device)
    N = neg.shape[0]
    P = pos.shape[0]
    
    # Default weights
    if weight_gen is None:
        weight_gen = torch.ones(G, device=gen.device)
    if weight_pos is None:
        weight_pos = torch.ones(P, device=gen.device)
    if weight_neg is None:
        weight_neg = torch.ones(N, device=gen.device)
    
    # 1. SCALING (адаптивное масштабирование как в JAX)
    targets = torch.cat([gen, neg, pos], dim=0)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=0)
    
    # Вычисляем масштаб на основе средневзвешенного расстояния
    dist_for_scale = torch.cdist(gen, targets)
    weighted_dist = dist_for_scale * targets_w.unsqueeze(0)  # [G, G+N+P]
    scale = weighted_dist.mean() / targets_w.mean()  # скаляр
    
    # Нормализуем координаты
    S = D  # размерность пространства
    scale_inputs = torch.clamp(scale / torch.sqrt(torch.tensor(S, dtype=torch.float32)), min=1e-3)
    gen_scaled = gen / scale_inputs
    targets_scaled = targets / scale_inputs
    
    # Нормализованное расстояние для ядра
    dist = torch.cdist(gen_scaled, targets_scaled)
    dist_normed = dist / torch.clamp(scale, min=1e-3)
    
    # 2. MASKING (self-взаимодействия)
    mask_val = 100.0
    diag_mask = torch.eye(G, device=gen.device)
    block_mask = F.pad(diag_mask, (0, N + P))  # [G, G+N+P]
    dist_normed = dist_normed + block_mask * mask_val
    
    # 3. MULTIPLE TEMPERATURES (усреднение по нескольким температурам)
    force_across_temp = torch.zeros_like(gen_scaled)
    
    for temp in temp_list:
        logits = -dist_normed / temp
        
        # 4. KERNEL NORMALIZATION (как в JAX)
        affinity = torch.softmax(logits, dim=-1)
        aff_transpose = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
        
        # Применяем веса
        affinity = affinity * targets_w.unsqueeze(0)  # [G, G+N+P]
        
        # Разделяем на отрицательные и положительные части
        split_idx = G + N
        aff_neg = affinity[:, :split_idx]      # [G, G+N]
        aff_pos = affinity[:, split_idx:]      # [G, P]
        
        # Вычисляем коэффициенты для силы
        sum_pos = aff_pos.sum(dim=-1, keepdim=True)  # [G, 1]
        r_coeff_neg = -aff_neg * sum_pos              # [G, G+N]
        sum_neg = aff_neg.sum(dim=-1, keepdim=True)   # [G, 1]
        r_coeff_pos = aff_pos * sum_neg               # [G, P]
        
        R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=1)  # [G, G+N+P]
        
        # Вычисляем силу для данной температуры
        total_force_temp = R_coeff @ targets_scaled  # [G, D]
        
        # Коррекция (гарантируем нулевую сумму коэффициентов)
        total_coeffs = R_coeff.sum(dim=-1, keepdim=True)  # [G, 1]
        total_force_temp = total_force_temp - total_coeffs * gen_scaled
        
        # Нормализуем силу для каждой температуры
        f_norm = (total_force_temp ** 2).mean()
        force_scale = torch.sqrt(torch.clamp(f_norm, min=1e-8))
        force_across_temp = force_across_temp + total_force_temp / force_scale
    
    # Возвращаем усредненную силу
    V = force_across_temp
    
    # Опционально: возвращаем дополнительную информацию
    # return V, {'scale': scale, 'scale_inputs': scale_inputs}
    return V


import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Вычисляет попарные евклидовы расстояния между векторами.
    
    Args:
        x: [B, N, D]
        y: [B, M, D]
        eps: небольшое значение для численной стабильности
    
    Returns:
        dist: [B, N, M]
    """
    # x^2 + y^2 - 2xy
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # [B, M, 1]
    xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
    
    sq_dist = x_sq + y_sq.transpose(1, 2) - 2 * xy
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    temp_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Drift loss для обучения с самоконтролем.
    
    Args:
        gen: [B, C, ...] - сгенерированные эмбеддинги
        fixed_pos: [B, C_p, ...] - положительные фиксированные эмбеддинги
        fixed_neg: [B, C_n, ...] - отрицательные фиксированные эмбеддинги (опционально)
        weight_gen: [B, C] - веса для gen (опционально)
        weight_pos: [B, C_p] - веса для fixed_pos (опционально)
        weight_neg: [B, C_n] - веса для fixed_neg (опционально)
        R_list: список температур для ядра
    
    Returns:
        loss: [B] - значение loss для каждого элемента в батче
        info: словарь с дополнительной информацией
    """
    
    # 1. Приводим входные данные к правильной размерности [B, C, S]
    # где S - размерность эмбеддинга (спейс)
    
    # Если вход [B, C, H, W], то сворачиваем пространственные размеры
    if gen.dim() == 4:
        B, C, H, W = gen.shape
        S = H * W
        gen = gen.view(B, C, S)
        if fixed_pos is not None:
            fixed_pos = fixed_pos.view(B, -1, S)
        if fixed_neg is not None:
            fixed_neg = fixed_neg.view(B, -1, S)
    elif gen.dim() == 2:
        # Если вход [B, C], то S = 1
        B, C = gen.shape
        S = 1
        gen = gen.unsqueeze(-1)
        if fixed_pos is not None:
            fixed_pos = fixed_pos.unsqueeze(-1)
        if fixed_neg is not None:
            fixed_neg = fixed_neg.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported input shape: {gen.shape}. Expected [B, C] or [B, C, H, W]")
    
    # 2. Проверяем размерности
    C_g = gen.shape[1]
    C_p = fixed_pos.shape[1]
    
    # Если fixed_neg не задан, создаем пустой тензор
    if fixed_neg is None:
        fixed_neg = torch.zeros(B, 0, S, device=gen.device, dtype=gen.dtype)
    C_n = fixed_neg.shape[1]
    
    # 3. Настройка весов
    if weight_gen is None:
        weight_gen = torch.ones(B, C_g, device=gen.device, dtype=gen.dtype)
    if weight_pos is None:
        weight_pos = torch.ones(B, C_p, device=gen.device, dtype=gen.dtype)
    if weight_neg is None:
        weight_neg = torch.ones(B, C_n, device=gen.device, dtype=gen.dtype)
    
    # 4. Приводим к float32
    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()
    
    # 5. Останавливаем градиенты для старого gen
    old_gen = gen.detach()
    
    # 6. Собираем targets
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)  # [B, C_g + C_n + C_p, S]
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)  # [B, C_g + C_n + C_p]
    
    # 7. Основная логика (без градиентов)
    with torch.no_grad():
        # Вычисляем расстояния
        dist = cdist(old_gen, targets)  # [B, C_g, C_g + C_n + C_p]
        
        # Взвешенное расстояние
        weighted_dist = dist * targets_w.unsqueeze(1)  # [B, C_g, C_g + C_n + C_p]
        scale = weighted_dist.mean(dim=(1, 2)) / targets_w.mean(dim=1)  # [B]
        
        info = {"scale": scale}
        
        # Нормализуем координаты
        scale_inputs = torch.clamp(scale / torch.sqrt(torch.tensor(S, dtype=torch.float32, device=gen.device)), 
                                   min=1e-3)
        old_gen_scaled = old_gen / scale_inputs.unsqueeze(-1).unsqueeze(-1)
        targets_scaled = targets / scale_inputs.unsqueeze(-1).unsqueeze(-1)
        
        # Нормализуем расстояния для ядра
        dist_normed = dist / torch.clamp(scale, min=1e-3).unsqueeze(-1).unsqueeze(-1)
        
        # Создаем маску для диагонали
        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=torch.float32)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))
        block_mask = block_mask.unsqueeze(0)  # [1, C_g, C_g + C_n + C_p]
        dist_normed = dist_normed + block_mask * mask_val
        
        # Вычисляем силу для каждой температуры
        force_across_R = torch.zeros_like(old_gen_scaled)
        
        for R in temp_list:
            logits = -dist_normed / R
            
            # Симметричная affinity
            affinity = F.softmax(logits, dim=-1)
            aff_transpose = F.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
            
            # Применяем веса
            affinity = affinity * targets_w.unsqueeze(1)
            
            # Разделяем на отрицательные и положительные
            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]
            
            # Вычисляем коэффициенты
            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg
            
            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)
            
            # Вычисляем силу
            total_force_R = torch.bmm(R_coeff, targets_scaled)
            
            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs.unsqueeze(-1) * old_gen_scaled
            
            f_norm_val = (total_force_R ** 2).mean(dim=(1, 2))  # [B]
            info[f"loss_{R}"] = f_norm_val
            
            # Нормализуем силу
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale.unsqueeze(-1).unsqueeze(-1)
        
        # Целевое значение в масштабированном пространстве
        goal_scaled = old_gen_scaled + force_across_R
    
    # 8. Вычисляем финальный loss с градиентами
    gen_scaled = gen / scale_inputs.unsqueeze(-1).unsqueeze(-1)

    return F.mse_loss(gen_scaled, goal_scaled)


def compute_drift_paper(x, y_pos, y_neg=None, T=0.05):
    """
    Реализация из Algorithm 2 статьи.
    x: [B, D] - сгенерированные образцы
    y_pos: [B, D] - положительные (реальные) образцы
    y_neg: [B, D] - отрицательные (другие сгенерированные), опционально
    """
    B, D = x.shape
    
    # Если отрицательные не заданы, используем x (как в статье)
    if y_neg is None:
        y_neg = x.clone()
    
    # Вычисляем расстояния
    dist_pos = torch.cdist(x, y_pos)  # [B, B]
    dist_neg = torch.cdist(x, y_neg)  # [B, B]
    
    # Маскируем диагональ для отрицательных (чтобы не сравнивать с собой)
    eye = torch.eye(B, device=x.device)
    dist_neg = dist_neg + eye * 1e6
    
    # Логиты
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T
    
    # Объединяем для softmax
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # [B, 2B]
    
    # Softmax по обеим осям (как в Algorithm 2)
    A_row = F.softmax(logit, dim=-1)
    A_col = F.softmax(logit, dim=-2)
    A = torch.sqrt(A_row * A_col + 1e-8)
    
    # Разделяем обратно
    A_pos, A_neg = torch.split(A, B, dim=1)  # [B, B], [B, B]
    
    # Вычисляем веса как в статье
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [B, B]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [B, B]
    
    # Дрейфующее поле
    V = (W_pos @ y_pos) - (W_neg @ y_neg)  # [B, D]
    
    return V