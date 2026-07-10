import torch
import torch.nn.functional as F


def compute_drift_single_temp(x, y_pos, y_neg=None, T=0.05):
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


def compute_drift(x, y_pos, y_neg=None, T_list=[0.02, 0.05, 0.2]):
    V_total = torch.zeros_like(x)
    
    if not isinstance(T_list, list):
        T_list = [T_list]

    for T in T_list:
        V = compute_drift_single_temp(x, y_pos, y_neg, T)
        V_total += V
    
    V_avg = V_total / len(T_list)
    
    return V_avg


def drifting_loss(
        x: torch.Tensor, 
        y_pos: torch.Tensor,
        y_neg: torch.Tensor = None, 
        compute_drift=compute_drift,
    ):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""

    x = x.float()
    y_pos = y_pos.float()
    y_neg = y_neg.float() if y_neg is not None else None

    with torch.no_grad():
        V = compute_drift(x, y_pos, y_neg)
        target = (x + V).detach()
    return F.mse_loss(x, target)


def drifting_loss_multifeatures(
        x: list, 
        y_pos: list,
        y_neg: list = None, 
        compute_drift=compute_drift,
    ):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    total_loss = 0
    n_steps = len(x)
    for step in range(n_steps):
        total_loss += drifting_loss(
            x[step],
            y_pos[step],
            y_neg[step] if y_neg is not None else None,
            compute_drift=compute_drift,
        )

    return total_loss
