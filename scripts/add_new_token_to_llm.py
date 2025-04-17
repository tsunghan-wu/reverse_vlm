import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="lmsys/vicuna-7b-v1.5", help="Name of the model to update")
    parser.add_argument("--output-dir", type=str, default="./updated_model", help="Directory to save the updated model")
    args = parser.parse_args()
    # Specify the model name
    model_name = args.model_name
    save_directory = args.output_dir
    os.makedirs(save_directory, exist_ok=True)

    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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

    # Save the updated tokenizer and model to a directory
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Updated tokenizer and model saved to {save_directory}.")
