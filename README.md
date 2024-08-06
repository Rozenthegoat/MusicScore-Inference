# MusicScore-Inference

Official inference script for text-driven Music Score Generation experiment of [paper](https://arxiv.org/abs/2406.11462):

**MusicScore: A Dataset for Music Score Modeling and Generation**.

> [Yuheng Lin](https://rozenthegoat.github.io), [Zheqi Dai](https://github.com/dzq84) and [Qiuqiang Kong](https://github.com/qiuqiangkong)

## Setup

This codebase is under Python 3.11. To setup a conda environment, run the following:

```
conda create -n ScoreGen python=3.11
conda activate ScoreGen
```

```
pip install -r requirement.yaml
```

## Run

An example shell script to run the inference is provided `inference.sh`.

```
bash inference.sh
```

It is also encouraged to make your own attempt by modifying the arguments:

```
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --prompt "a music score of piano" \
  --ckpt_path "checkpoint/epoch_78000.pt" \
  --config "config.yaml" \
  --outpath "result.jpg" \
  --n_sample 5 \
  --resolution 512 \
  --cfg_scale 4.0 \
  --ddim_steps 10 \
```

## Gradio Demo

We also support a simple Gradio web interface for easier inference.

```
python gradio_demo.py
```

By testing, inferencing 512 resolution of 9 images at one time consumes 23.58G GPU memory, within the capacity of one RTX 4090.
