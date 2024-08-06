import gradio as gr
from omegaconf import OmegaConf
from datetime import datetime
from rich import print
import numpy as np
from PIL import Image
from einops import rearrange
import torch
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
from utils import count_params, instantiate_from_config


def generate_image(
    prompt, ckpt_path, config_path, resolution, n_sample, cfg_scale, ddim_steps, outpath
):
    device = torch.device("cuda")

    config = OmegaConf.load(config_path)
    print(f"Loading model by config: {config_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    if state_dict.get("model_state_dict") is not None:
        if state_dict.get("global_step") is not None:
            gs = state_dict.get("global_step")
            print(
                f"Inference ScoreDiffusion at global step: {state_dict['global_step']}"
            )
        state_dict = state_dict["model_state_dict"]
    else:
        state_dict = state_dict

    model = instantiate_from_config(config.model)

    print(f"Start loading model weight...")
    m, u = model.load_state_dict(state_dict, strict=False)
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)
    print(f"Finish loading Stable Diffusion.")

    count_params(model, verbose=True)
    model.to(device)

    C = config.model.params.channels  # num of latent channel
    H, W = resolution, resolution
    f = 8
    shape = [C, H // f, W // f]
    sampler = DDIMSampler(model, device=device)

    log = f"""
DDIM Sampler Configuration:
C: {C}
H: {H} W: {W}
k: {f}
shape: {shape}
ddim_steps: {ddim_steps}
Number of samples: {n_sample}
Model global step: {gs}
time: {datetime.now()}"""
    print(log)

    model.eval()
    with torch.no_grad():
        text = n_sample * [prompt]

        c = model.get_learned_conditioning(text)
        uc = model.get_learned_conditioning(len(text) * [""])

        z_denoise, intermediates = sampler.sample(
            S=ddim_steps,
            batch_size=n_sample,
            shape=shape,
            conditioning=c,
            verbose=False,
            log_every_t=50,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uc,
        )

        output = model.decode_first_stage(z_denoise)
        output = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)

        image_list = []

        grid_count = 0
        for x_sample in output:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            img = Image.fromarray(x_sample.astype(np.uint8))
            image_list.append(img)
            grid_count += 1

        grid_image = make_grid(
            output, nrow=int(np.ceil(len(image_list) ** 0.5)), normalize=True
        )
        grid_image = grid_image.cpu().permute(1, 2, 0).numpy()
        grid_image = (grid_image * 255).astype(np.uint8)
        grid_image = Image.fromarray(grid_image)

        grid_image.save(outpath)
        print(f"Reconstructed image saved to {outpath}")

        return grid_image


def main():
    interface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="Prompt", value="a violin score, key is A major"),
            gr.Textbox(label="Checkpoint Path", value="/path/to/ckpt"),
            gr.Textbox(label="Config Path", value="config.yaml"),
            gr.Slider(128, 1024, step=64, label="Resolution", value="512"),
            gr.Slider(1, 16, step=1, label="Number of Samples", value="9"),
            gr.Slider(0.0, 10.0, step=0.1, label="CFG Scale", value="4.0"),
            gr.Slider(1, 1000, step=1, label="DDIM Steps", value="250"),
            gr.Textbox(label="Output Path", value="result.jpg"),
        ],
        outputs=gr.Image(label="Generated Image"),
    )
    interface.launch()


if __name__ == "__main__":
    main()
