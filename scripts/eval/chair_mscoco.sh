#!/bin/bash
# LLaVA-v1.5-7B
CKPT_PATH="tsunghanwu/reverse_llava_v15"
MODE="llava_v15"

# LLaVA-MORE (LLaVA w/ Llama-3.1-Instruct-8B)
# CKPT_PATH="tsunghanwu/reverse_llava_more"
# MODE="llama31"

CKPT_SUFFIX=""  # useful for debugging
UN_THRESHOLD=0.003
IS_OPEN_ENDED_QA="False"
MSCOCO_DIR="./playground/data/eval/chair_mscoco"


if [ "$MODE" = "llava_v15" ]; then
    LLM_SETTING="--conv-mode vicuna_v1 --llm_backbone vicuna1.5_7b"
elif [ "$MODE" = "llama31" ]; then
    LLM_SETTING="--conv-mode llama_3_1 --llm_backbone llama_3_1 --llm_pad_token pad"
else
    echo "Invalid mode: $MODE"
    exit
fi

#########################################################################################################
# Do not modify below this line
export TOKENIZER_PATH=$CKPT_PATH

# GPU settings
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# CKPT is the last part of the path, turn it into a shell variable
CKPT="${CKPT_PATH##*/}${CKPT_SUFFIX}"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --dataset "MSCOCO-CHAIR" \
        --question-file "${MSCOCO_DIR}/chair-500.jsonl" \
        --image-folder "${MSCOCO_DIR}/coco/val2014" \
        --answers-file "${MSCOCO_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        $LLM_SETTING \
        --un_threshold $UN_THRESHOLD \
        --open_ended_qa $IS_OPEN_ENDED_QA &

done

wait

output_file="${MSCOCO_DIR}/answers/$CKPT/merge.jsonl"
output_file=$(realpath "$output_file")
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${MSCOCO_DIR}/answers/$CKPT/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python3 llava/eval/eval_chair.py \
    --coco_path "${MSCOCO_DIR}/coco/annotations/" \
    --cache "${MSCOCO_DIR}/chair.pkl" \
    --cap_file $output_file \
    --save_path "${MSCOCO_DIR}/answers/${CKPT}_eval_log.json" \
