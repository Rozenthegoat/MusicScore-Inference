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
