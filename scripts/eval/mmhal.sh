#!/bin/bash
# LLaVA-v1.5-7B
CKPT_PATH="tsunghanwu/reverse_llava_v15"
MODE="llava_v15"

# LLaVA-MORE (LLaVA w/ Llama-3.1-Instruct-8B)
# CKPT_PATH="tsunghanwu/reverse_llava_more"
# MODE="llama31"

# Qwen2.5-VL-3B
# CKPT_PATH="tsunghanwu/reverse_qwen25_vl"
# MODE="qwen25_vl"

CKPT_SUFFIX=""
UN_THRESHOLD=0.003
IS_OPEN_ENDED_QA="True"
MMHAL_DIR="./playground/data/eval/mmhal"

if [ "$MODE" = "llava_v15" ]; then
    LLM_SETTING="--conv-mode vicuna_v1 --llm_backbone vicuna1.5_7b"
    EXEC="llava.eval.model_vqa_loader"
elif [ "$MODE" = "llama31" ]; then
    LLM_SETTING="--conv-mode llama_3_1 --llm_backbone llama_3_1 --llm_pad_token pad"
    EXEC="llava.eval.model_vqa_loader"
elif [ "$MODE" = "qwen25_vl" ]; then
    LLM_SETTING=""
    EXEC="qwenvl.eval.model_vqa_loader"
else
    echo "Invalid mode: $MODE"
    exit
fi

export OPENAI_API_KEY="sk-***" # for evaluation purposes only

#########################################################################################################
# Do not modify below this line

export TOKENIZER_PATH=$CKPT_PATH
CKPT="${CKPT_PATH##*/}${CKPT_SUFFIX}"

# GPU settings
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $EXEC \
        --model-path $CKPT_PATH \
        --question-file Shengcao1006/MMHal-Bench \
        --answers-file "${MMHAL_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" \
        --dataset "MMHAL" \
        $LLM_SETTING \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --un_threshold $UN_THRESHOLD \
        --open_ended_qa $IS_OPEN_ENDED_QA &

done

wait

output_file="${MMHAL_DIR}/answers/${CKPT}/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${MMHAL_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

# Evaluation
# sleep for safety
sleep 2

python scripts/eval/eval_mmhal.py \
    --question-file Shengcao1006/MMHal-Bench \
    --result-file $output_file \
    --log_file "${MMHAL_DIR}/answers/${CKPT}.log" \
    --gpt-model gpt-4-0314
