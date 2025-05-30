import os
import argparse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Name of the model to update")
    parser.add_argument("--output-dir", type=str, default="./updated_qwen25_vlm_3b_new", help="Directory to save the updated model")
    args = parser.parse_args()
    # Specify the model name
    model_name = args.model_name
    save_directory = args.output_dir
    os.makedirs(save_directory, exist_ok=True)

    # Load the pre-trained tokenizer and model
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)

    # Define new tokens to be added
    new_tokens = ["<SPAN>", "</CN>", "</UN>"]

    # Check which tokens are already in the vocabulary
    tokens_to_add = [token for token in new_tokens if token not in tokenizer.get_vocab()]
    assert len(tokens_to_add) == len(new_tokens), "Some tokens are already in the vocabulary."

    # Add new tokens to the tokenizer
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
        # Resize the model's embeddings to accommodate the new tokens
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {len(tokens_to_add)} tokens to the tokenizer and resized model embeddings.")
    else:
        print("No new tokens to add.")
    token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    for token, token_id in zip(new_tokens, token_ids):
        print(f"Token: {token}, ID: {token_id}")
    # Save the updated tokenizer and model to a directory
    processor.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Saved updated processor and model to {save_directory}")