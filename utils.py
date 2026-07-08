import numpy as np
import torch
import matplotlib.pyplot as plt


def get_number_of_parameters(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    print("Params:", n_params)


def show_batch_grid(batch, n=4, figsize=(12, 12), denormalize=True):
    """
    Display batch of images as a grid without any labels or numbering.
    
    Args:
        batch (torch.Tensor): Batch of images of shape (B, C, H, W)
        n (int): Number of images per row/column (grid will be n x n)
        figsize (tuple): Figure size (width, height) in inches
        denormalize (bool): Apply denormalization if images are in [-1, 1]
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
        
        plt.tight_layout()
        plt.show()