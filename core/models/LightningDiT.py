import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights
from timm.models.vision_transformer import PatchEmbed, Mlp

from core.models.swiglu_ffn import SwiGLUFFN 
from core.models.pos_embed import VisionRotaryEmbeddingFast
from core.models.rmsnorm import RMSNorm


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LightningDiTConfig:
    """
    Configuration class for LightningDiT model.
    """
    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        num_transformer_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        use_checkpointing: bool = False,
        **kwargs
    ):
        self.model_type = "LightningDiT"
        self.dtype = dtype
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_qknorm = use_qknorm
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_checkpointing = use_checkpointing
        
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
    def from_pretrained(cls, path: str) -> "LightningDiTConfig":
        """
        Load configuration from JSON file.
        """
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Verify model type
        if config_dict.get("model_type") != "LightningDiT":
            raise ValueError(f"Expected model_type 'LightningDiT', got '{config_dict.get('model_type')}'")
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "LightningDiTConfig":
        """
        Create config from dictionary.
        """
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return f"LightningDiTConfig({self.to_json_string()})"


class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            

    # @torch.compile
    def forward(self, x, feat_rope=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    # @torch.compile
    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config: LightningDiTConfig
    ):
        super().__init__()
        self.config = config

        self.input_size = config.input_size
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads
        self.mlp_ratio = config.mlp_ratio
        self.use_qknorm = config.use_qknorm
        self.use_swiglu = config.use_swiglu
        self.use_rope = config.use_rope
        self.use_rmsnorm = config.use_rmsnorm
        self.num_transformer_blocks = config.num_transformer_blocks
        self.hidden_size = config.hidden_size
        self.use_checkpointing = config.use_checkpointing
        
        self.x_embedder = PatchEmbed(
            self.input_size, 
            self.patch_size, 
            self.in_channels, 
            self.hidden_size, 
            bias=True
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = self.hidden_size // self.num_heads // 2
            hw_seq_len = self.input_size // self.patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size=self.hidden_size, 
                num_heads=self.num_heads, 
                mlp_ratio=self.mlp_ratio, 
                use_qknorm=self.use_qknorm, 
                use_swiglu=self.use_swiglu, 
                use_rmsnorm=self.use_rmsnorm,
                ) for _ in range(self.num_transformer_blocks)
        ])
        self.final_layer = FinalLayer(
            self.hidden_size, 
            self.patch_size, 
            self.out_channels, 
            use_rmsnorm=self.use_rmsnorm
        )
        self.initialize_weights()
        self.to(getattr(torch, config.dtype))


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.1)  # <<< малая дисперсия
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    def forward(self, x):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        for block in self.blocks:
            if self.use_checkpointing:
                x = checkpoint(block, x, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, self.feat_rope)

        x = self.final_layer(x)     # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)      # (N, out_channels, H, W)

        return x


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
    def from_pretrained(path, device="cpu", dtype=torch.bfloat16, verbose=False):
        """
        Load a pretrained LightningDiT model from the specified path.
        
        Args:
            path (str): Directory path where the model is saved
            
        Returns:
            LightningDiT: Loaded model with pretrained weights
        """
        # Load configuration
        config = LightningDiTConfig.from_pretrained(path)
        
        # Create model with loaded config
        with init_empty_weights():
            model = LightningDiT(config=config)
        model = model.to_empty(device=device)
        model = model.to(getattr(torch, config.dtype))
        
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

