import os
import json
import argparse
import shortuuid
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, rejection_sampling_generate
from llava.mm_utils import get_model_name_from_path
from llava.eval.data_utils import get_dataloader
from llava.eval.inference_utils import retrospective_resampling_generate, OPEN_ENDED_QA_DEFAULT_ANSWER


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


def set_llama31_pad_token(tokenizer, args):
    if args.llm_pad_token == 'end_of_text':
        tokenizer.pad_token_id = 128001
    elif args.llm_pad_token == 'eot':
        tokenizer.pad_token_id = 128009
    elif args.llm_pad_token == 'pad':
        tokenizer.pad_token_id = 128004
    else:
        raise ValueError(f"Unknown llm_pad_token")
    print(f"pad token: {args.llm_pad_token}")



def eval_model(args):
    # Initialization: Model, Tokenizer, Image Processor
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    if args.llm_backbone == "llama_3_1":
        set_llama31_pad_token(tokenizer, args)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # Preparing the dataset and dataloader
    data_loader = get_dataloader(args, tokenizer, image_processor, model.config)

    # Create the output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for (input_ids, image_tensor, image_sizes, metadata) in tqdm(data_loader, total=len(data_loader)):
        metadata = metadata[0]
        question_id = metadata["question_id"]
        image_id = metadata["image_id"]
        cur_prompt = metadata["question"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        # retrospective resampling
        input_data = (model, tokenizer, input_ids, image_tensor, image_sizes, args)
        decoded_text, generation_history, successful_correction = retrospective_resampling_generate(
            *input_data, add_open_ended_qa_prompt=False
        )
        if args.open_ended_qa and decoded_text == "":
            decoded_text, generation_history, successful_correction = retrospective_resampling_generate(
                *input_data, add_open_ended_qa_prompt=True
            )
            if decoded_text == "":
                decoded_text = OPEN_ENDED_QA_DEFAULT_ANSWER
        
        write_result(question_id, cur_prompt, image_id, decoded_text, generation_history, successful_correction, ans_file, model_name)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model settings
    parser.add_argument("--model-path", type=str, default="tsunghanwu/reverse_llava_v15")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--llm_backbone", type=str, default="vicuna1.5_7b")
    parser.add_argument("--llm_pad_token", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")

    # Data settings
    parser.add_argument("--dataset", type=str, default="MSCOCO-CHAIR", choices=["MSCOCO-CHAIR", "AMBER-G", "AMBER-D", "Haloquest", "POPE", "MME", "MMHAL"])
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl", help="Path to save the answers")

    # Basic Generation settings
    parser.add_argument("--temperature", type=float, default=1.0)
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
