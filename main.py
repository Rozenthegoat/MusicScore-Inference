from omegaconf import OmegaConf
from datetime import datetime
from rich import print
import numpy as np
from PIL import Image
from einops import rearrange
import argparse
import torch
from torchvision.utils import make_grid

from ldm.models.diffusion.ddim import DDIMSampler
from utils import count_params, instantiate_from_config


def main(args):

    device = torch.device("cuda")

    config = OmegaConf.load(args.config)
    print(f"Loading model by config: {args.config}")
    state_dict = torch.load(args.ckpt_path, map_location="cpu")

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
    H, W = args.resolution, args.resolution
    f = 8
    shape = [C, H // f, W // f]
    ddim_steps = args.ddim_steps
    sampler = DDIMSampler(model, device=device)

    log = f"""
DDIM Sampler Configuration:
C: {C}
H: {H} W: {W}
k: {f}
shape: {shape}
ddim_steps: {ddim_steps}
Number of samples: {args.n_sample}
Model global step: {gs}
time: {datetime.now()}"""
    print(log)

    model.eval()
    with torch.no_grad():
        prompt = args.prompt
        text = args.n_sample * [prompt]

        c = model.get_learned_conditioning(text)
        uc = model.get_learned_conditioning(len(text) * [""])

        z_denoise, intermediates = sampler.sample(
            S=ddim_steps,
            batch_size=args.n_sample,
            shape=shape,
            conditioning=c,
            verbose=False,
            log_every_t=50,
            unconditional_guidance_scale=args.cfg_scale,
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

        grid_image.save(args.outpath)
        print(f"Reconstructed image saved to {args.outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a music score, instrumentation is violin, key is A major",
        help="Text prompt input",
    )
    parser.add_argument("--ckpt_path", type=str, help="Model checkpoint path")
    parser.add_argument(
        "--config", type=str, help="Model config path for Stable Diffusion 2.0 model"
    )
    parser.add_argument("--outpath", type=str, help="Score generation save path")
    parser.add_argument(
        "--n_sample",
        type=int,
        help="Number of samples to generate conditioned on current prompt",
    )
    parser.add_argument(
        "--resolution", type=int, help="Resolution of the generated score image"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for sampling, 4.0 by default",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=250,
        help="DDIM sampler denoising steps, 250 by default",
    )

    args = parser.parse_args()
    main(args)

    print(
        f"Maximal memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
    )
