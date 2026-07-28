"""
MAE-ResNet Feature Encoder for CelebA (128x128)
Based on the official JAX implementation from "Generative Modeling via Drifting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


def safe_std(x: torch.Tensor, dim: Union[int, List[int]], eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    """Безопасное вычисление стандартного отклонения"""
    mean = x.mean(dim=dim, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.maximum(var, torch.tensor(eps, device=x.device)))


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Преобразует вход в патчи"""
    if patch_size == 1:
        return x
    
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 3, 5, 2, 4)  # (B, C, patch_size, patch_size, H//patch_size, W//patch_size)
    return x.reshape(B, C * patch_size * patch_size, H // patch_size, W // patch_size)


def unpatchify(x: torch.Tensor, patch_size: int, original_channels: int) -> torch.Tensor:
    """Восстанавливает изображение из патчей"""
    if patch_size == 1:
        return x
    
    B, C_total, H_patches, W_patches = x.shape
    C = original_channels
    
    # Обратный порядок действий
    x = x.reshape(B, C, patch_size, patch_size, H_patches, W_patches)
    x = x.permute(0, 1, 4, 2, 5, 3)  # (B, C, H_patches, patch_size, W_patches, patch_size)
    return x.reshape(B, C, H_patches * patch_size, W_patches * patch_size)


def choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


class BasicBlock(nn.Module):
    def __init__(self, filters: int, in_channels: Optional[int] = None,
                 stride: int = 1, gn_max_groups: int = 32, dropout_prob: float = 0.0):
        super().__init__()
        # ВАЖНО: если in_channels не указан, используем filters
        if in_channels is None:
            in_channels = filters
            
        self.in_channels = in_channels
        self.filters = filters
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(choose_gn_groups(filters, gn_max_groups), filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(choose_gn_groups(filters, gn_max_groups), filters)
        self.drop = nn.Dropout(dropout_prob)

        # Проекция нужна если изменились размеры или каналы
        if stride != 1 or in_channels != filters:
            self.proj_conv = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
            self.proj_gn = nn.GroupNorm(choose_gn_groups(filters, gn_max_groups), filters)
        else:
            self.proj_conv = None
            self.proj_gn = None

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if self.proj_conv is not None:
            nn.init.kaiming_normal_(self.proj_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        y = self.drop(y) if train else y
        y = self.conv2(y)
        y = self.gn2(y)

        if self.proj_conv is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_gn(residual)

        return F.relu(residual + y)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
        in_channels: int = 3,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.in_channels = in_channels

        # Stem
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(choose_gn_groups(base_channels, gn_max_groups), base_channels)

        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()

        # Каналы на каждом этапе
        stage_channels = []
        for stage_idx in range(len(layers)):
            if stage_idx == 0:
                stage_channels.append(base_channels)
            else:
                stage_channels.append(base_channels * (2 ** stage_idx))

        for stage_idx, num_blocks in enumerate(layers):
            out_ch = stage_channels[stage_idx]
            stride = 2 if stage_idx > 0 else 1
            
            # Входные каналы для первого блока
            if stage_idx == 0:
                in_ch = base_channels  # после conv1
            else:
                in_ch = stage_channels[stage_idx - 1]

            blocks = []
            # Первый блок может иметь stride=2
            blocks.append(
                BasicBlock(
                    filters=out_ch,
                    in_channels=in_ch,
                    stride=stride,
                    gn_max_groups=gn_max_groups,
                    dropout_prob=dropout_prob,
                )
            )
            # Остальные блоки с stride=1
            for _ in range(1, num_blocks):
                blocks.append(
                    BasicBlock(
                        filters=out_ch,
                        in_channels=out_ch,  # Входные каналы равны выходным
                        stride=1,
                        gn_max_groups=gn_max_groups,
                        dropout_prob=dropout_prob,
                    )
                )

            self.stages.append(nn.Sequential(*blocks))
            self.stage_norms.append(
                nn.GroupNorm(choose_gn_groups(out_ch, gn_max_groups), out_ch)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(
        self,
        x: torch.Tensor,
        train: bool = True,
        return_block_outputs: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]]:
        feats = {}
        block_outputs = {}

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        feats['conv1'] = x

        for i, (stage, norm) in enumerate(zip(self.stages, self.stage_norms)):
            layer_name = f'layer{i+1}'
            for block_id, block in enumerate(stage):
                x = block(x, train=train)
                if return_block_outputs:
                    block_outputs[f"layer_name_{block_id}"] = x
            x = norm(x)
            feats[layer_name] = x

        if return_block_outputs:
            return feats, block_outputs
        return feats


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2, bias=False)
        self.gn = nn.GroupNorm(choose_gn_groups(out_channels, 32), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        concat_channels = in_channels + out_channels
        self.concat_norm = nn.GroupNorm(choose_gn_groups(concat_channels, 32), concat_channels)
        self.proj = ConvGNReLU(concat_channels, out_channels, kernel=3)
        self.refine = ConvGNReLU(out_channels, out_channels, kernel=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c1 = base_channels          # 64
        c2 = base_channels          # 64
        c3 = base_channels * 2      # 128
        c4 = base_channels * 4      # 256
        c5 = base_channels * 8      # 512

        self.bridge = ConvGNReLU(c5, c5, kernel=3)
        self.up43 = UpBlock(in_channels=c5, out_channels=c4)
        self.up32 = UpBlock(in_channels=c4, out_channels=c3)
        self.up21 = UpBlock(in_channels=c3, out_channels=c2)
        self.up10 = UpBlock(in_channels=c2, out_channels=c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.bridge(feats['layer4'])
        x = self.up43(x, feats['layer3'])
        x = self.up32(x, feats['layer2'])
        x = self.up21(x, feats['layer1'])
        x = self.up10(x, feats['conv1'])
        return self.head(x)


class MAEResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        mask_patch_size: int = 2,
        input_patch_size: int = 8, # = 1 if for latent representations
        dropout_prob: float = 0.0,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3), # ResNet50
        num_classes: int = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.mask_patch_size = mask_patch_size
        self.input_patch_size = input_patch_size
        self.dropout_prob = dropout_prob
        self.layers = layers
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = ResNetEncoder(
            base_channels=base_channels,
            layers=layers,
            dropout_prob=dropout_prob,
            in_channels=in_channels * input_patch_size * input_patch_size,
        )
        
        # Decoder
        self.decoder = UNetDecoder(
            base_channels=base_channels,
            out_channels=in_channels * input_patch_size * input_patch_size,
        )
        
        # Classification head
        if self.num_classes is not None: 
            self.fc = nn.Linear(base_channels * 8, num_classes)  # layer4: base_channels * 8
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def make_patch_mask(
        self,
        x: torch.Tensor,
        mask_ratio: torch.Tensor,
        patch_size: int = 2,
    ) -> torch.Tensor:
        """Создает бинарную маску для патчей"""
        B, _, H, W = x.shape
        nh, nw = H // patch_size, W // patch_size
        
        # Создаем шум для маскирования
        noise = torch.rand(B, nh, nw, device=x.device, dtype=x.dtype)
        
        # Маска: 1 - замаскировано, 0 - видимо
        mask = (noise < mask_ratio).to(x.dtype)
        
        # Upsample до размера изображения
        mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.50,
        labels: Optional[torch.Tensor] = None,
        lambda_cls: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Patch input
        x = patchify(x, self.input_patch_size)
        
        # Masking
        mask = self.make_patch_mask(x, mask_ratio, self.mask_patch_size)
        x_masked = x * (1.0 - mask)
        
        # Encoder
        feats = self.encoder(x_masked, train=True)

        if (labels is not None) and (self.num_classes is not None):
            # Global pooling & classification
            top = feats['layer4']
            pooled = top.mean(dim=(2, 3))
            logits = self.fc(pooled)
        
        # Decoder
        recon = self.decoder(feats)
        recon_loss = F.mse_loss(recon * mask, x * mask)
        
        metrics = {
            'recon_loss': recon_loss,
        }
        
        if (labels is not None) and (self.num_classes is not None):
            cls_loss = F.cross_entropy(logits, labels, reduction='none')
            metrics['cls_loss'] = cls_loss
            metrics['accuracy'] = (logits.argmax(dim=-1) == labels).float()
            loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss
        else:
            loss = recon_loss
        
        metrics['loss'] = loss
        return loss, metrics
    

class FeatureExtractor(nn.Module):
    """
    Обертка для извлечения признаков из MAE-ResNet.
    Используется в Drifting Model.
    """
    
    def __init__(
        self,
        encoder: ResNetEncoder,
        input_patch_size: int = 8,
        use_scale: bool = True,
        patch_mean_size: List[int] = [2, 4],
        patch_std_size: List[int] = [2, 4],
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.input_patch_size = input_patch_size
        self.use_scale = use_scale
        self.patch_mean_size = patch_mean_size
        self.patch_std_size = patch_std_size
        self.use_std = use_std
        self.use_mean = use_mean
        self.every_k_block = every_k_block
    

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Извлекает признаки для Drifting Model.
        
        Args:
            x: входные изображения (B, C, H, W)
        
        Returns:
            Dict[str, torch.Tensor]: словарь признаков, каждый формы (B, T, D)
        """
        x = patchify(x, self.input_patch_size)
        
        # Проход через энкодер с промежуточными выходами
        need_blocks = (
            isinstance(self.every_k_block, (int, float)) and 
            not math.isinf(float(self.every_k_block)) and 
            self.every_k_block >= 1
        )
        
        if need_blocks:
            feats, block_outputs = self.encoder(x, train=False, return_block_outputs=True)
        else:
            feats = self.encoder(x, train=False)
            block_outputs = {}
        feats.update(block_outputs)

        features_by_dim = {}
        
        if self.use_scale:
            input_stats = (x ** 2).mean(dim=[2, 3])  # [B, 3]
            features_by_dim.setdefault(3, []).append(input_stats.unsqueeze(0))
        
        for f_name, f in feats.items():
            B_f, C, Hf, Wf = f.shape
            
            # --- (a) Пространственные векторы ---
            spatial = f.permute(0, 2, 3, 1).reshape(B_f, Hf * Wf, C)  # [B, N, C]
            spatial_features = spatial.permute(1, 0, 2)  # [N, B, C]
            features_by_dim.setdefault(C, []).append(spatial_features)
            
            # --- (b) Глобальные статистики ---
            mean_global = f.mean(dim=[2, 3])  # [B, C]
            std_global = f.std(dim=[2, 3])    # [B, C]
            features_by_dim.setdefault(C, []).append(mean_global.unsqueeze(0))  # [1, B, C]
            features_by_dim.setdefault(C, []).append(std_global.unsqueeze(0))    # [1, B, C]
            
            # --- (c) Патчи 2×2 ---
            if Hf >= 2 and Wf >= 2:
                mean_p2, std_p2 = self._get_patch_stats(f, 2)
                features_by_dim.setdefault(C, []).append(mean_p2.permute(1, 0, 2))  # [num_patches, B, C]
                features_by_dim.setdefault(C, []).append(std_p2.permute(1, 0, 2))
            
            # --- (d) Патчи 4×4 ---
            if Hf >= 4 and Wf >= 4:
                mean_p4, std_p4 = self._get_patch_stats(f, 4)
                features_by_dim.setdefault(C, []).append(mean_p4.permute(1, 0, 2))
                features_by_dim.setdefault(C, []).append(std_p4.permute(1, 0, 2))
        
        # Конкатенация по размерности N_features
        grouped_features = []
        for dim, feature_list in features_by_dim.items():
            stacked = torch.cat(feature_list, dim=0)  # [total_N_features, B, D]
            grouped_features.append(stacked)
        
        return grouped_features

    @torch.compile()
    def _get_patch_stats(self, f, patch_size):
        """Возвращает среднее и std для каждого патча размера patch_size."""
        B, C, H, W = f.shape
        
        patches = f.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches: [B, C, num_h, num_w, patch_size, patch_size]
        
        mean_patch = patches.mean(dim=[-2, -1])  # [B, C, num_h, num_w]
        var_patch = patches.var(dim=[-2, -1])    # [B, C, num_h, num_w]
        std_patch = torch.sqrt(var_patch + 1e-8)  # [B, C, num_h, num_w]
        
        # Превращаем в [B, num_patches, C]
        mean_patch = mean_patch.permute(0, 2, 3, 1).reshape(B, -1, C)
        std_patch = std_patch.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        return mean_patch, std_patch


    # def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     """
    #     Извлекает признаки для Drifting Model.
        
    #     Args:
    #         x: входные изображения (B, C, H, W)
        
    #     Returns:
    #         Dict[str, torch.Tensor]: словарь признаков, каждый формы (B, T, D)
    #     """
    #     # Базовые признаки
    #     out: Dict[str, torch.Tensor] = {}
        
    #     # Масштаб
    #     if self.use_scale:
    #         out['norm_x'] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[None, :, :]
        
    #     # MAE признаки
    #     mae_feats = self.get_activations(
    #         x,
    #         patch_mean_size=self.patch_mean_size,
    #         patch_std_size=self.patch_std_size,
    #         use_std=self.use_std,
    #         use_mean=self.use_mean,
    #         every_k_block=self.every_k_block,
    #     )
    #     out.update(mae_feats)
        
    #     return out


    # def get_activations(
    #     self,
    #     x: torch.Tensor,
    #     patch_mean_size: Optional[List[int]] = None,
    #     patch_std_size: Optional[List[int]] = None,
    #     use_std: bool = True,
    #     use_mean: bool = True,
    #     every_k_block: float = 2.0,
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Извлекает многоуровневые признаки из энкодера для Drifting Model.
        
    #     Args:
    #         x: входные изображения (B, C, H, W)
    #         patch_mean_size: размеры патчей для вычисления среднего
    #         patch_std_size: размеры патчей для вычисления std
    #         use_std: использовать ли std
    #         use_mean: использовать ли mean
    #         every_k_block: частота извлечения промежуточных блоков
        
    #     Returns:
    #         Dict[str, torch.Tensor]: словарь признаков. Каждое значение имеет форму (B, T, D),
    #         где T - количество пространственных токенов, D - размерность каналов.
    #     """
    #     patch_mean_size = patch_mean_size or []
    #     patch_std_size = patch_std_size or []
        
    #     # Patch input
    #     x = patchify(x, self.input_patch_size)
        
    #     # Проход через энкодер с промежуточными выходами
    #     need_blocks = (
    #         isinstance(every_k_block, (int, float)) and 
    #         not math.isinf(float(every_k_block)) and 
    #         every_k_block >= 1
    #     )
        
    #     if need_blocks:
    #         feats, block_outputs = self.encoder(x, train=False, return_block_outputs=True)
    #     else:
    #         feats = self.encoder(x, train=False)
    #         block_outputs = {}
        
    #     out: Dict[str, torch.Tensor] = {}
        
    #     # Нормализованный x (для информации о масштабе)
    #     out['norm_x'] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[None, :, :]
        
    #     def process_feat(name: str, feat: torch.Tensor) -> None:
    #         """Обрабатывает один feature map"""
    #         B, C, H, W = feat.shape
            
    #         # Основной признак: [B, H*W, C]
    #         # out[name] = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
    #         out[name] = feat.permute(2, 3, 0, 1).reshape(-1, B, C)
            
    #         # Глобальные статистики
    #         if use_mean:
    #             out[f'{name}_mean'] = feat.mean(dim=(2, 3))[None, :, :]
    #         if use_std:
    #             out[f'{name}_std'] = safe_std(feat, dim=(2, 3))[None, :, :]
            
    #         # Статистики по патчам
    #         for size in patch_mean_size:
    #             if H % size == 0 and W % size == 0:
    #                 reshaped = feat.unfold(2, size, size).unfold(3, size, size)
    #                 reshaped = reshaped.contiguous().view(B, C, H // size, W // size, size * size)
    #                 mean_patches = reshaped.mean(dim=-1)
    #                 # out[f'{name}_mean_{size}'] = mean_patches.permute(0, 2, 3, 1).reshape(B, -1, C)
    #                 out[f'{name}_mean_{size}'] = mean_patches.permute(2, 3, 0, 1).reshape(-1, B, C)
            
    #         for size in patch_std_size:
    #             if H % size == 0 and W % size == 0:
    #                 reshaped = feat.unfold(2, size, size).unfold(3, size, size)
    #                 reshaped = reshaped.contiguous().view(B, C, H // size, W // size, size * size)
    #                 std_patches = safe_std(reshaped, dim=-1)
    #                 # out[f'{name}_std_{size}'] = std_patches.permute(0, 2, 3, 1).reshape(B, -1, C)
    #                 out[f'{name}_std_{size}'] = std_patches.permute(2, 3, 0, 1).reshape(-1, B, C)
        
    #     # Обработка всех feature maps
    #     for name, feat in feats.items():
    #         process_feat(name, feat)
        
    #     # Промежуточные блоки
    #     if need_blocks:
    #         k = int(every_k_block)
    #         for i in range(1, 5):
    #             lname = f'layer{i}'
    #             blocks = block_outputs.get(lname, [])
    #             for blk_idx, feat_i in enumerate(blocks, start=1):
    #                 if blk_idx % k == 0:
    #                     process_feat(f'{lname}_blk{blk_idx}', feat_i)
        
    #     return out