#!/bin/bash

# 开启显存碎片整理，防止 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动训练
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset medical-o1-reasoning-SFT \
    --template deepseek \
    --finetuning_type lora \
    --lora_target all \
    --output_dir results/MedCoT-7B-Final \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --quantization_bit 4 \
    --plot_loss True \
    --fp16 True \
    --seed 42