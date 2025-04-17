#!/bin/bash
# LLaVA-v1.5-7B
CKPT_PATH="tsunghanwu/reverse_llava_v15"
MODE="llava_v15"

# LLaVA-MORE (LLaVA w/ Llama-3.1-Instruct-8B)
# CKPT_PATH="tsunghanwu/reverse_llava_more"
# MODE="llama31"

CKPT_SUFFIX=""  # useful for debugging
UN_THRESHOLD=0.5
IS_OPEN_ENDED_QA="False"
MME_DIR="./playground/data/eval/MME"

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
# CKPT is the last part of the path, turn it into a shell variable
CKPT="${CKPT_PATH##*/}${CKPT_SUFFIX}"

# GPU settings
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --dataset "MME" \
        --question-file "lmms-lab/MME" \
        --answers-file "${MME_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        $LLM_SETTING \
        --un_threshold $UN_THRESHOLD \
        --open_ended_qa $IS_OPEN_ENDED_QA &

done

wait

output_file="${MME_DIR}/answers/${CKPT}/merge.jsonl"
output_file=$(realpath "$output_file")
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${MME_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python llava/eval/eval_mme.py --pred_file $output_file --results_dir "${MME_DIR}/eval_tool/answers/${CKPT}"
