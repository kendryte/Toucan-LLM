#!/bin/bash

torchrun --nproc_per_node=4 --master_port=8080 train.py \
    --model_name_or_path llama_to_hf_path \
    --data_path data_path \
    --bf16 True \
    --output_dir model_save_path \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 8e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True

