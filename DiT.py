import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """SwiGLU FFN LightningDiT"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        
    def forward(self, x):
        # SwiGLU: x * w1(x) * sigmoid(w3(x))
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.mlp = FeedForward(dim, int(dim * mlp_ratio))
        
    def forward(self, x):
        # Attention
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm, x_norm, x_norm)[0]
        x = x + x_attn
        
        # MLP
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp
        return x


class TinyDiT(nn.Module):
    def __init__(
        self,
        image_size=32,
        in_channels=1,
        out_channels=1,
        patch_size=2,  # 32/2 = 16x16 токенов → 256 токенов
        dim=128,       # Маленькая размерность для игрушечной модели
        num_heads=8,
        n_blocks=6,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Токенизация (превращаем патчи в векторы)
        self.input_proj = nn.Linear(
            in_channels * patch_size * patch_size, 
            dim
        )
        
        # Позиционное кодирование (используем learnable, но можно RoPE)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, dim) * 0.02
        )
        
        # Блоки DiT
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads)
            for _ in range(n_blocks)
        ])
        
        # Финальная нормализация и выход
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_proj = nn.Linear(dim, out_channels * patch_size * patch_size)
        

    def forward(self, x):
        """
        x: [B, C, H, W] — зашумленное изображение
        y: [B] — метки классов (опционально)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Разбиваем на патчи: [B, C, H, W] -> [B, num_patches, p*p*C]

        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, self.num_patches, -1)
        
        # Проецируем в скрытое пространство
        x = self.input_proj(x) + self.pos_embed
        

        # Проходим через блоки DiT
        for block in self.blocks:
            x = block(x)
        
        # Финальный выход
        x = self.norm_out(x)
        x = self.output_proj(x)
        
        # Собираем обратно в изображение
        x = x.view(B, H // p, W // p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        
        return x

# # Использование
# model = TinyDiT(
#     image_size=64,
#     in_channels=1,
#     out_channels=1,
#     patch_size=2,  # 32/2 = 16x16 токенов → 256 токенов
#     dim=128,       # Маленькая размерность для игрушечной модели
#     num_heads=8,
#     n_blocks=6
# )

# # Пример forward
# x = torch.randn(4, 1, 64, 64)
# output = model(x)
# print(output.shape)  # [4, 3, 32, 32]