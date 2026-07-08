import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(residual + h)


class TinyVAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, base_channels=64): # Увеличил базу до 64
        super().__init__()
        
        # ============ ENCODER ============
        self.enc_conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc_norm1 = nn.GroupNorm(8, base_channels)
        
        self.enc_conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.enc_norm2 = nn.GroupNorm(8, base_channels * 2)
        
        # Downsample
        self.enc_down = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.enc_norm_down = nn.GroupNorm(8, base_channels * 2)
        
        self.enc_res = ResidualBlock(base_channels * 2)
        
        # Выход в латентное пространство
        self.enc_out = nn.Conv2d(base_channels * 2, latent_channels * 2, 3, padding=1)
        
        # ============ DECODER ============
        self.dec_in = nn.Conv2d(latent_channels, base_channels * 2, 3, padding=1)
        self.dec_norm_in = nn.GroupNorm(8, base_channels * 2)
        
        self.dec_res = ResidualBlock(base_channels * 2)
        
        # FIX 1: Upsample + Conv вместо ConvTranspose2d (убираем шахматку)
        self.dec_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        
        self.dec_conv1 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.dec_norm1 = nn.GroupNorm(8, base_channels)
        
        self.dec_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        
        self.latent_dim = latent_channels

    def encode(self, x):
        h = F.silu(self.enc_norm1(self.enc_conv1(x)))
        h = F.silu(self.enc_norm2(self.enc_conv2(h)))
        h = F.silu(self.enc_norm_down(self.enc_down(h)))
        h = self.enc_res(h)
        h = self.enc_out(h)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = F.silu(self.dec_norm_in(self.dec_in(z)))
        h = self.dec_res(h)
        h = self.dec_up(h)  # <-- Исправленный апсемплинг
        h = F.silu(self.dec_norm1(self.dec_conv1(h)))
        # h = torch.sigmoid(self.dec_out(h))
        h = self.dec_out(h)
        return h

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar, z


def vae_loss(recon, x, mean, logvar, beta=0.0005):
    recon_loss = F.mse_loss(recon, x, reduction='mean')  # x в [0,1], recon в [0,1]
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(
    dataloader: DataLoader,
    vae: nn.Module,
    epochs: int = 100,
    lr: float = 1e-4,
    beta: float = 1.0,
    device: str = 'cuda',
    save_dir: str = 'checkpoints'
):
    vae = vae.to(device)
    optimizer = AdamW(vae.parameters(), lr=lr)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            x = batch.to(device)
            
            optimizer.zero_grad()

            recon, mean, logvar, _ = vae(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mean, logvar, beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        print(f'Epoch {epoch+1}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}')
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'vae_epoch_{epoch+1}.pt'))
            
            # Save sample reconstructions
            vae.eval()
            with torch.no_grad():
                sample = x[:8]
                recon_sample, _, _, _ = vae(sample)
                torch.save({
                    'original': sample.cpu(),
                    'reconstructed': recon_sample.cpu()
                }, os.path.join(save_dir, f'sample_epoch_{epoch+1}.pt'))
    
    torch.save(vae.state_dict(), os.path.join(save_dir, 'vae_final.pt'))
    return vae
