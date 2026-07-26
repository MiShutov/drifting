import os
from itertools import cycle
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from training.utils import show_batch_grid
from training.drifting import (
    drifting_loss_multifeatures
)


def prepare_vae(
    vae_name,
    path_to_vae,
    img_size,
    vae_kwargs
):
    if vae_name == "flux_vae":
        from diffusers import AutoencoderKLFlux2
        vae =  AutoencoderKLFlux2.from_pretrained(
            path_to_vae,
            **vae_kwargs
        ).eval()

        latent_shape = (32, img_size//8, img_size//8)
        return vae, latent_shape


def train_drifting(
        dataset,
        feature_encoder,
        transformer,
        opt,
        noise_shape,
        vae=None,
        T_list=[0.02, 0.05, 0.2],
        batch_size=64,
        gradient_accumulation=1,
        training_steps=10000,
        plot_every=250,
        seed=None,
        save_step=5000,
        loss_scaling=1.0,
        training_device="cuda:0",
        save_path="checkpoints/",
    ):
    """Train drifting model. Returns model and loss history."""
    os.makedirs(save_path, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=min(64, training_steps // gradient_accumulation),
        num_training_steps=training_steps // gradient_accumulation,
    )

    dataloader = cycle(DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    ))

    loss_history = []
    plot_loss_acc = []
    ema = None
    pbar = tqdm(range(1, training_steps + 1))
    
    for step in pbar:
        pos = next(dataloader).to(training_device)

        with torch.no_grad():
            if vae is not None:
                pos_latent = vae.encode(pos).latent_dist.sample()
                pos_features = feature_encoder(pos_latent)
            else:
                pos_features = feature_encoder(pos)

        noise = torch.randn(
            batch_size, 
            *noise_shape, 
            device=training_device, 
            dtype=pos.dtype
        )        
        
        gen_latent = transformer(noise)
        gen_features = feature_encoder(gen_latent)

        loss, logs = drifting_loss_multifeatures(
            x_features=gen_features,
            y_pos_features=pos_features,
            T_list=T_list,
        )
        loss = loss * loss_scaling

        loss.backward()
        if step % gradient_accumulation == 0:
            clip_grad_norm_(transformer.parameters(), max_norm=2.0)
            opt.step()
            scheduler.step()
            opt.zero_grad()

        V_abs = 0
        for k,v in logs.items():
            V_abs += v["V_abs"]

        plot_loss_acc.append(V_abs)
        ema = V_abs if ema is None else 0.96 * ema + 0.04 * V_abs
        pbar.set_postfix(loss=f"{ema:.2e}")

        if step % plot_every == 0 or step == 1:
            with torch.no_grad():
                noise = torch.randn(
                    18, 
                    *noise_shape,
                    device=training_device, 
                    dtype=pos.dtype
                )
                if vae is not None:
                    gen_latent = transformer(noise)
                    gen = vae.decode(gen_latent).sample
                else:
                    gen = transformer(noise)
    
                clear_output(wait=True)
                show_batch_grid(gen, n=6, figsize=(9, 9), denormalize=True, save_path=f"{save_path}/gen_images_{step}.png")
    
                loss_history.append(sum(plot_loss_acc) / len(plot_loss_acc))
                plot_loss_acc = []
                plt.semilogy(loss_history)
                plt.show()
            
                for _, log_i in logs.items():
                    for k,v in log_i.items():
                        print(k, v)

                # print()
                # for k,v in logs[4].items():
                #     print(k, v)

        if (step+1) % save_step == 0:
            os.makedirs(save_path, exist_ok=True)
            transformer.save_pretrained(f"{save_path}/transformer_step{step+1}")
            # torch.save(transformer.state_dict(), f"{save_path}/transformer_step{step+1}.pth")

    return loss_history
