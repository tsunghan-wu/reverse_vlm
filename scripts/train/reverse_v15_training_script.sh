#!/bin/bash
LLM_PATH="./vicuna1.5_7b_with_new_tokens"   # path to the LLM checkpoint
MODE="llava_v15"                          # mode to run the script
RUN_NAME="reverse_v15_7b"  # name of the run
DATA_PATH="./final_dataset.json"  # path to the data
PROJECTOR_PATH="checkpoints/llava_v15_pretraining/mm_projector.bin"  # path to the projector
export TOKENIZER_PATH=$LLM_PATH            # path to the tokenizer



if [ "$MODE" = "llava_v15" ]; then
    LLM_SETTING="--llm_backbone vicuna1.5_7b --version v1"
elif [ "$MODE" = "llama31" ]; then
    LLM_SETTING="--llm_backbone llama_3_1 --llm_pad_token pad --version llama_3_1"
else
    echo "Invalid mode: $MODE"
    exit
fi

# Local setting
GPU_SETTINGS="localhost:0,1,2,3,4,5,6,7"
MASTER_PORT="19487"

# Optional flags
export NCCL_P2P_DISABLE=0   # Enable P2P (NVLink, PCIe)
export NCCL_IB_DISABLE=0     # Enable InfiniBand (if available)
export NCCL_NET_GDR_LEVEL=2

deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/train/zero3.json \
    --model_name_or_path $LLM_PATH \
    $LLM_SETTING \
    --data_path $DATA_PATH \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $PROJECTOR_PATH \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --do_dehallucination_training True

