#!/bin/bash
# LLaVA-v1.5-7B
CKPT_PATH="tsunghanwu/reverse_llava_v15"
MODE="llava_v15"

# LLaVA-MORE (LLaVA w/ Llama-3.1-Instruct-8B)
# CKPT_PATH="tsunghanwu/reverse_llava_more"
# MODE="llama31"

CKPT_SUFFIX=""
UN_THRESHOLD=0.5
IS_OPEN_ENDED_QA="True"
POPE_DIR="./playground/data/eval/pope"

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
        --question-file "${POPE_DIR}/llava_pope_test.jsonl" \
        --image-folder "${POPE_DIR}/coco/val2014" \
        --answers-file "${POPE_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        $LLM_SETTING \
        --un_threshold $UN_THRESHOLD \
        --open_ended_qa $IS_OPEN_ENDED_QA &

done

wait

output_file="${POPE_DIR}/answers/${CKPT}/merge.jsonl"
output_file=$(realpath "$output_file")
echo "Output file: $output_file"

# Clear out the output file if it exists.
> "$output_file"

# # # # Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${POPE_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
 
# Evaluation
# sleep for safety
sleep 2


python llava/eval/eval_pope.py \
    --annotation-dir "${POPE_DIR}/coco_pope/" \
    --question-file "${POPE_DIR}/llava_pope_test.jsonl" \
    --result-file "$output_file"
