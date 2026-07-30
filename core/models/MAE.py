"""
MAE-ResNet Feature Encoder for CelebA (128x128)
Based on the official JAX implementation from "Generative Modeling via Drifting"
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from accelerate import init_empty_weights


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


class MAEResNetConfig:
    """
    Configuration class for MAEResNet model.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        mask_patch_size: int = 2,
        mask_ratio: float = 0.5,
        per_channel_mask: bool = False,
        input_patch_size: int = 1,
        dropout_prob: float = 0.0,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
        num_classes: int = None,
        **kwargs
    ):
        self.model_type = "MAEResNet"
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.per_channel_mask = per_channel_mask
        self.input_patch_size = input_patch_size
        self.dropout_prob = dropout_prob
        self.layers = layers
        self.num_classes = num_classes
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """
        Convert config to dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                config_dict[key] = value
        return config_dict
    
    def to_json_string(self) -> str:
        """
        Serialize config to JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def save_pretrained(self, path: str):
        """
        Save configuration to JSON file.
        """
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return config_path
    
    @classmethod
    def from_pretrained(cls, path: str) -> "MAEResNetConfig":
        """
        Load configuration from JSON file.
        """
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Verify model type
        if config_dict.get("model_type") != "MAEResNet":
            raise ValueError(f"Expected model_type 'MAEResNet', got '{config_dict.get('model_type')}'")
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "MAEResNetConfig":
        """
        Create config from dictionary.
        """
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return f"MAEResNetConfig({self.to_json_string()})"


class MAEResNet(nn.Module):
    def __init__(
        self,
        config: MAEResNetConfig
    ):
        super().__init__()
        self.config = config
        
        self.in_channels = config.in_channels
        self.base_channels = config.base_channels
        self.mask_patch_size = config.mask_patch_size
        self.mask_ratio = config.mask_ratio
        self.per_channel_mask = config.per_channel_mask
        self.input_patch_size = config.input_patch_size
        self.dropout_prob = config.dropout_prob
        self.layers = config.layers
        self.num_classes = config.num_classes
        
        # Encoder
        self.encoder = ResNetEncoder(
            base_channels=config.base_channels,
            layers=config.layers,
            dropout_prob=config.dropout_prob,
            in_channels=config.in_channels * config.input_patch_size * config.input_patch_size,
        )
        
        # Decoder
        self.decoder = UNetDecoder(
            base_channels=config.base_channels,
            out_channels=config.in_channels * config.input_patch_size * config.input_patch_size,
        )
        
        # Classification head
        if self.num_classes is not None: 
            self.fc = nn.Linear(config.base_channels * 8, config.num_classes)  # layer4: base_channels * 8
        
        self._initialize_weights()
        self.to(getattr(torch, config.dtype))

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def make_patch_mask(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.per_channel_mask:
            noise = torch.rand(B, C, H // self.mask_patch_size, W // self.mask_patch_size, device=x.device, dtype=x.dtype)
        else:
            noise = torch.rand(B, 1, H // self.mask_patch_size, W // self.mask_patch_size, device=x.device, dtype=x.dtype)
        
        mask = (noise < self.mask_ratio).to(x.dtype)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        return mask

    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lambda_cls: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Patch input
        x = patchify(x, self.input_patch_size)
        
        # Masking
        mask = self.make_patch_mask(x)
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


    def save_pretrained(self, path, verbose=False):
        """
        Save model weights and configuration to the specified path.
        Creates the directory if it doesn't exist.
        
        Args:
            path (str): Directory path where to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration using the config object
        config_path = self.config.save_pretrained(path)
        
        # Save model weights
        weights_path = os.path.join(path, "pytorch_model.pth")
        torch.save(self.state_dict(), weights_path)

        if verbose:
            print(f"Model saved to {path}")
            print(f"  - Config: {config_path}")
            print(f"  - Weights: {weights_path}")
            return config_path, weights_path


    @staticmethod
    def from_pretrained(path, device="cpu", dtype=None, verbose=False):
        """
        Load a pretrained MAEResNet model from the specified path.
        
        Args:
            path (str): Directory path where the model is saved
            
        Returns:
            MAEResNet: Loaded model with pretrained weights
        """
        # Load configuration
        config = MAEResNetConfig.from_pretrained(path)
        
        # Create model with loaded config
        with init_empty_weights():
            model = MAEResNet(config=config)
        model = model.to_empty(device=device)
        if dtype is None:
            dtype = getattr(torch, config.dtype)
        model = model.to(dtype)

        # Load weights
        weights_path = os.path.join(path, "pytorch_model.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

        if verbose:
            print(f"Model loaded from {path}")
            print(f"  - Config: {os.path.join(path, 'config.json')}")
            print(f"  - Weights: {weights_path}")
        
        return model

    

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
            
            # --- Spatial features ---
            spatial = f.permute(0, 2, 3, 1).reshape(B_f, Hf * Wf, C)  # [B, N, C]
            spatial_features = spatial.permute(1, 0, 2)  # [N, B, C]
            features_by_dim.setdefault(C, []).append(spatial_features)
            
            # --- Global stats ---
            mean_global = f.mean(dim=[2, 3])  # [B, C]
            std_global = f.std(dim=[2, 3])    # [B, C]
            features_by_dim.setdefault(C, []).append(mean_global.unsqueeze(0))  # [1, B, C]
            features_by_dim.setdefault(C, []).append(std_global.unsqueeze(0))    # [1, B, C]
            
            # --- Patches 2×2 ---
            if Hf >= 2 and Wf >= 2:
                mean_p2, std_p2 = self._get_patch_stats(f, 2)
                features_by_dim.setdefault(C, []).append(mean_p2)  # [num_patches, B, C]
                features_by_dim.setdefault(C, []).append(std_p2)
            
            # --- Patches 4×4 ---
            if Hf >= 4 and Wf >= 4:
                mean_p4, std_p4 = self._get_patch_stats(f, 4)
                features_by_dim.setdefault(C, []).append(mean_p4)
                features_by_dim.setdefault(C, []).append(std_p4)
        
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
        
        # [num_patches, B, C]
        mean_patch = mean_patch.permute(2, 3, 0, 1).reshape(-1, B, C)
        std_patch = std_patch.permute(2, 3, 0, 1).reshape(-1, B, C)
        return mean_patch, std_patch

