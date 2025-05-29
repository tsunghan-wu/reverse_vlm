import os
import json
import torch
import argparse
import shortuuid
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .data_utils import get_dataloader

def write_result(idx, cur_prompt, image_id, decoded_text, generation_history, successful_correction, ans_file, model_name):
    ans_id = shortuuid.uuid()
    if not isinstance(image_id, str):
        image_id = ""
    ans_file.write(json.dumps({"question_id": idx,
                            "prompt": cur_prompt,
                            "image_id": image_id,
                            "text": decoded_text.strip(),
                            "generation_history": generation_history,
                            "answer_id": ans_id,
                            "model_id": model_name,
                            "successful_correction": successful_correction,
                            "metadata": {}}) + "\n")
    ans_file.flush()


def eval_model(args):
    # Initialization: Model, Tokenizer, Image Processor
    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model = model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # Preparing the dataset and dataloader
    data_loader = get_dataloader(args, processor, model.config)

    # Create the output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    with torch.inference_mode():
        for inputs, metadata in tqdm(data_loader, total=len(data_loader)):
            metadata = metadata[0]
            question_id = metadata["question_id"]
            image_id = metadata["image_id"]
            cur_prompt = metadata["question"]
            inputs = inputs[0].to(device='cuda', non_blocking=True)
            # retrospective resampling
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            decoded_text = [text.replace("<SPAN>", "").replace("</CN>", "").replace("</UN>", "") for text in decoded_text]
            generation_history = []
            successful_correction = True
            write_result(question_id, cur_prompt, image_id, decoded_text[0], generation_history, successful_correction, ans_file, model_path)
            torch.cuda.empty_cache()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model settings
    parser.add_argument("--model-path", type=str, default="tsunghanwu/reverse_llava_v15")
    # Data settings
    parser.add_argument("--dataset", type=str, default="MSCOCO-CHAIR", choices=["MSCOCO-CHAIR", "AMBER-G", "AMBER-D", "Haloquest", "POPE", "MME", "MMHAL"])
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl", help="Path to save the answers")

    # Basic Generation settings
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    # Retrospective Resampling settings
    parser.add_argument("--un_threshold", type=float, default=0.003)
    parser.add_argument("--temperature_increase", type=float, default=0.1)
    parser.add_argument("--max_total_generation_attempts", type=int, default=50)
    parser.add_argument("--max_local_generation_attempts", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Print detailed information during evaluation")
    # We handle open-ended QA and captioning in a slightly different way for query-rewriting
    parser.add_argument("--open_ended_qa", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
