import os
from itertools import cycle
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from core.image_utils import show_batch_grid, compute_FID
from core.training.drifting import (
    drifting_loss_multifeatures
)


def validation(
    step,
    save_path,
    batch_size,
    noise_shape,
    transformer,
    vae=None,
    noise_dtype=torch.bfloat16,
    device="cuda:0",
    samples_to_compute_fid=None,
    real_dataloader=None,
):
    with torch.no_grad():
        title = f"Generated images step={step}"
        if samples_to_compute_fid is not None:
            n_batches = samples_to_compute_fid // batch_size
            # Prepare real images
            real_images = []
            for _ in tqdm(range(n_batches)):
                batch = next(real_dataloader)
                real_images.append(batch)
            real_images = torch.cat(real_images, dim=0)

            # Prepare generated images
            gen_images = []
            for _ in tqdm(range(n_batches)):
                noise = torch.randn(batch_size, *noise_shape, dtype=torch.bfloat16, device=device)
                with torch.no_grad():
                    gen_latent = transformer(noise)
                    gen_images.append(
                        vae.decode(gen_latent).sample.detach().to(torch.float32).cpu()
                    )
            gen_images = torch.cat(gen_images, dim=0)
            fid_score = compute_FID(real_images, gen_images)
            title += f" FID={fid_score:g}"

        # Plot generated samples
        noise = torch.randn(batch_size, *noise_shape, device=device, dtype=noise_dtype)
        if vae is not None:
            gen_latent = transformer(noise)
            gen = vae.decode(gen_latent).sample
        else:
            gen = transformer(noise)

        clear_output(wait=True)
        show_batch_grid(
            gen[:18], 
            n=6, 
            figsize=(9, 9), 
            denormalize=True, 
            save_path=f"{save_path}/gen_images_{step}.png",
            title=title
        )



def train_drifting(
        dataset,
        feature_encoder,
        transformer,
        opt,
        noise_shape,
        vae=None,
        T_list=[0.02, 0.05, 0.2],
        samples_block=None,
        batch_size=64,
        gradient_accumulation=1,
        training_steps=10000,
        val_step=1000,
        samples_to_compute_fid=4096,
        seed=None,
        save_step=5000,
        loss_scaling=1.0,
        training_device="cuda:0",
        training_dtype=torch.bfloat16,
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

    real_dataloader = cycle(DataLoader(
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
        pos = next(real_dataloader).to(training_dtype).to(training_device)

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
            samples_block=samples_block,
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

        if step % val_step == 0 or step == 1:
            validation(
                step,
                save_path,
                batch_size,
                noise_shape,
                transformer,
                vae,
                noise_dtype=training_dtype,
                device=training_device,
                samples_to_compute_fid=samples_to_compute_fid,
                real_dataloader=real_dataloader,
            )
            loss_history.append(sum(plot_loss_acc) / len(plot_loss_acc))
            plot_loss_acc = []
            plt.semilogy(loss_history)
            plt.show()

        if (step+1) % save_step == 0:
            os.makedirs(save_path, exist_ok=True)
            transformer.save_pretrained(f"{save_path}/transformer_step{step+1}")

    return loss_history
