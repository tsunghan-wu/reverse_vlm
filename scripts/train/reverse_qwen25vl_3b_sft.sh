#!/bin/bash

MODEL_NAME="./qwen25_vlm_3b_with_new_tokens"  # path to the model checkpoint
DATA_PATH="./final_dataset_subsample.json"  # path to the data (we FT Qwen2.5-VL on a subset of the full dataset)
RUN_NAME="reverse_qwen25vl_3b"  # name of the run
# GPU Setting
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
GPU_SETTINGS="localhost:4,5,6,7"
MASTER_PORT="15999"


PYTHONPATH=. deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT qwenvl/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/train/zero3.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder ./playground/data \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-5 \
    --merger_lr 5e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --do_dehallucination_training True
