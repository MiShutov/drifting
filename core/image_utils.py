from tqdm.auto import tqdm
import torch
from torchmetrics.image.fid import FrechetInceptionDistance as FID
import numpy as np
import matplotlib.pyplot as plt


def normalize_to_uint8(images):
    """
    images: torch.Tensor [-1, 1], float32
    returns: torch.Tensor [0, 255], uint8
    """
    images = (images + 1) * 127.5
    images = torch.clamp(images, 0, 255)
    images = images.to(torch.uint8)
    return images


def compute_FID(real_images, gen_images, batch_size=128, device="cuda:0"):
    metric_fid = FID().to(device)
    n_real_batches = real_images.shape[0] // batch_size
    n_gen_batches = gen_images.shape[0] // batch_size

    for b_id in tqdm(range(n_real_batches)):
        metric_fid.update(normalize_to_uint8(
            real_images[batch_size*b_id:batch_size*(b_id+1)]).to(device), real=True
        )

    for b_id in tqdm(range(n_gen_batches)):
        metric_fid.update(
            normalize_to_uint8(gen_images[batch_size*b_id:batch_size*(b_id+1)]).to(device), real=False
        )

    fid_score = metric_fid.compute()
    return fid_score


def show_batch_grid(
        batch, 
        n=4, 
        figsize=(12, 12), 
        denormalize=True, 
        save_path=None, 
        title=None):
    """
    Display batch of images as a grid without any labels or numbering.
    
    Args:
        batch (torch.Tensor): Batch of images of shape (B, C, H, W)
        n (int): Number of images per row/column (grid will be n x n)
        figsize (tuple): Figure size (width, height) in inches
        denormalize (bool): Apply denormalization if images are in [-1, 1]
        save_path (str): Path to save the image. If None, only displays.
    """
    with torch.no_grad():
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().detach().float().numpy()
        
        # Handle dimensions: (B, C, H, W) -> (B, H, W, C)
        if batch.ndim == 4 and batch.shape[1] in [1, 3]:
            if batch.shape[1] == 1:
                batch = batch.squeeze(axis=1)  # (B, H, W)
            else:
                batch = np.transpose(batch, (0, 2, 3, 1))  # (B, H, W, C)
        
        # Denormalize if needed
        if denormalize:
            batch = np.clip(batch * 0.5 + 0.5, 0, 1)
        
        # Limit to n*n images
        n_images = min(n * n, len(batch))
        batch = batch[:n_images]
        
        # Create grid
        plt.figure(figsize=figsize)

        for i in range(n_images):
            plt.subplot(n, n, i + 1)
            img = batch[i]
            
            if img.ndim == 2:
                plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(img)
            
            plt.axis('off')


        if title:
            plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.96])  # rect=[left, bottom, right, top]
        else:
            plt.tight_layout(pad=0.1)

        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        if title:
            plt.suptitle(title, fontsize=int(figsize[0]*1.6))
    

        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            print(f"Saved image to {save_path}")
        
        plt.show()