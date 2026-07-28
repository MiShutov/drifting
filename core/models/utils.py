def get_number_of_parameters(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    
    if n_params >= 1e9:
        print(f"Params: {n_params / 1e9:.3g}B")
    elif n_params >= 1e6:
        print(f"Params: {n_params / 1e6:.3g}M")
    else:
        print(f"Params: {n_params:,}")


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