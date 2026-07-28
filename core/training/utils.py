import torch


def prepare_noise(batch_size, noise_shape, device="cuda:0", dtype=torch.bfloat16, initial_seed=None):
    """Prepare noise for generation."""
    if initial_seed is not None:
        torch.manual_seed(initial_seed)
    return torch.randn(batch_size, *noise_shape, device=device, dtype=dtype)
