#!/bin/bash
# LLaVA-v1.5-7B
CKPT_PATH="tsunghanwu/reverse_llava_v15"
MODE="llava_v15"

# LLaVA-MORE (LLaVA w/ Llama-3.1-Instruct-8B)
# CKPT_PATH="tsunghanwu/reverse_llava_more"
# MODE="llama31"

CKPT_SUFFIX=""
UN_THRESHOLD=0.003
IS_OPEN_ENDED_QA="False"
HALOQUEST_DIR="./playground/data/eval/haloquest"

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


# TODO: We should add default answers here rather than in the eval script
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --dataset "HALOQUEST" \
        --question-file "${HALOQUEST_DIR}/haloquest-eval.jsonl" \
        --image-folder "${HALOQUEST_DIR}/eval_images" \
        --answers-file "${HALOQUEST_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" \
        $LLM_SETTING \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --un_threshold $UN_THRESHOLD \
        --open_ended_qa $IS_OPEN_ENDED_QA &

done

wait

output_file="${HALOQUEST_DIR}/answers/${CKPT}/merge.jsonl"
output_file=$(realpath "$output_file")
echo "Output file: $output_file"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${HALOQUEST_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
 
# Evaluation
# sleep for safety
sleep 2

# Evaluation code: (hard to combine that into a single conda environment...)
export GOOGLE_API_KEY="***"

python3 llava/eval/eval_haloquest.py \
    --question-file "${HALOQUEST_DIR}/haloquest-eval.jsonl" \
    --result-file $output_file \
    --evaluation-result-file "${output_file%.jsonl}.log"