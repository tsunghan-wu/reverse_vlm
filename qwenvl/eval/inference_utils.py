import re
import torch
import string


GENERAL_QUERY_REWRITING_PROMPT = "(Hint: potential incorrect phrases -> {})"
OPEN_ENDED_QA_QUERY_REWRITING_PROMPT = "For this question, please point out the false premises or note what information is missing, rather than answering it directly."
OPEN_ENDED_QA_DEFAULT_ANSWER = "Unable to answer directly due to false premises or missing information."


def check_special_token_mismatch(span_indices, un_indices, cn_indices):
    """
    Check if the number of special tokens (</UN>, </CN>, <SPAN>) match in the output.
    """
    open_tags = span_indices
    close_tags = torch.cat([un_indices, cn_indices], dim=0)
    close_tags, _ = close_tags.sort()
    if open_tags.shape[0] != close_tags.shape[0]:
        return False
    for open_tag, close_tag in zip(open_tags, close_tags):
        if open_tag > close_tag:
            return False
    return True


def render_output(output_ids_squeezed, tokenizer, special_indices, cut_current_sentence=False):
    """
    Remove special tokens from the output for readability.
    """
    mask = torch.ones_like(output_ids_squeezed, dtype=torch.bool)
    mask[special_indices] = False
    refined_output_ids = output_ids_squeezed[mask].unsqueeze(0)
    decoded_output = tokenizer.batch_decode(refined_output_ids, skip_special_tokens=True)[0].strip()
    # Reattach punctuation to the previous word.
    # This regex finds any whitespace preceding a punctuation character and removes that space.
    readable_output = re.sub(r'\s([%s])' % re.escape(string.punctuation), r'\1', decoded_output)
    # assert type(readable_output) == str, f"Readable output is not a string: {readable_output}"
    if isinstance(readable_output, bytes):
        readable_output = readable_output.decode("utf-8")
    if cut_current_sentence:
        # Cut the current sentence
        sentences = readable_output.split(".")
        if len(sentences) > 1:
            readable_output = ". ".join(sentences[:-1]) + "."
        else:
            readable_output = ""
    return readable_output


def de_hallucination_postprocessing(existing_output_ids, new_output_ids, scores, tokenizer, args, skip_first=False, correction_mode=False):
    hallucination_phrase = None
    back_track_idx = -1
    if correction_mode:
        # we only need to check until the first </CN> token for the new_ouptut_ids
        un_token_id = tokenizer.convert_tokens_to_ids("</UN>")
        cn_token_id = tokenizer.convert_tokens_to_ids("</CN>")
        span_token_id = tokenizer.convert_tokens_to_ids("<SPAN>")
        un_indices = torch.where(new_output_ids.squeeze(0) == un_token_id)[0]
        cn_indices = torch.where(new_output_ids.squeeze(0) == cn_token_id)[0]
        closing_indices = torch.cat([un_indices, cn_indices], dim=0)
        closing_indices, _ = closing_indices.sort()
        if len(closing_indices) > 0:
            new_output_ids = new_output_ids[:, :closing_indices[0] + 1]
            new_output_ids = torch.concat([new_output_ids, torch.tensor([[tokenizer.eos_token_id]], device=new_output_ids.device)], dim=1)
            list_scores = list(scores)
            scores = tuple(list_scores[:closing_indices[0] + 1] + [list_scores[-1]])
        # print("Correction mode -> Truncation", flush=True)
    latest_idx = new_output_ids.shape[1]
    output_ids = torch.cat([existing_output_ids, new_output_ids], dim=1)
    # 1. Raw decoding
    raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if args.verbose:
        print("-" * 50)
        print(f"Raw output: {raw_output}")
        print("-" * 50)

    # 2. Sanity check for special tokens
    assert len(output_ids) == 1, "Batch size must be 1."
    output_ids_squeezed = output_ids.squeeze(0)

    # Find out those special tokens
    un_token_id = tokenizer.convert_tokens_to_ids("</UN>")
    cn_token_id = tokenizer.convert_tokens_to_ids("</CN>")
    span_token_id = tokenizer.convert_tokens_to_ids("<SPAN>")

    span_indices = torch.where(output_ids_squeezed == span_token_id)[0]
    un_indices = torch.where(output_ids_squeezed == un_token_id)[0]
    cn_indices = torch.where(output_ids_squeezed == cn_token_id)[0]
    if len(span_indices) == 0:
        # No hallucination detected
        # Naive way to remove special tokens
        raw_output = raw_output.split("</UN>")[0].replace("</CN>", "").strip()
        return raw_output, raw_output, new_output_ids, hallucination_phrase, back_track_idx, latest_idx

    # 3. Remove all special tokens (</UN>, </CN>, and <SPAN>) from output_ids for readability.
    special_indices = torch.cat([span_indices, un_indices, cn_indices], dim=0)
    readable_output = render_output(output_ids_squeezed, tokenizer, special_indices)
    # Ensure the opening and closing tags match.
    if check_special_token_mismatch(span_indices, un_indices, cn_indices) is False:
        if args.verbose:
            print("Mismatched special token counts detected. --> Regenerating.")
            print(raw_output, flush=True)
        back_track_idx = 0
        hallucination_phrase = "Format Error"
        return raw_output, readable_output, new_output_ids, hallucination_phrase, back_track_idx, latest_idx

    # 4. Check Hallucination
    # 4(a). Get the probabilities of tokens
    logit_scores = torch.stack(scores, dim=1).squeeze(0)  # shape: (sequence_length, vocab_size)
    logit_probs = torch.nn.functional.softmax(logit_scores, dim=-1)

    closing_indices = torch.cat([un_indices, cn_indices], dim=0)
    NP_probs = []
    # 4(b). Check the probability of UN token for each NP
    new_output_ids_squeezed = new_output_ids.squeeze(0)
    span_indices = torch.where(new_output_ids_squeezed == span_token_id)[0]
    un_indices = torch.where(new_output_ids_squeezed == un_token_id)[0]
    cn_indices = torch.where(new_output_ids_squeezed == cn_token_id)[0]
    closing_indices = torch.cat([un_indices, cn_indices], dim=0)
    closing_indices, _ = closing_indices.sort()

    for span_idx, close_idx in zip(span_indices, closing_indices):
        phrase_tokens = new_output_ids_squeezed[span_idx + 1:close_idx]
        phrase = tokenizer.decode(phrase_tokens, skip_special_tokens=True).strip()

        closing_token_prob = logit_probs[close_idx, un_token_id].item()
        closing_token_cn_prob = logit_probs[close_idx, cn_token_id].item()
        NP_probs.append((phrase, closing_token_prob, closing_token_cn_prob, span_idx, close_idx))
    # visualize the NP_probs
    for idx, entry in enumerate(NP_probs):
        phrase, prob, cn_prob, span_idx, close_idx = entry
        if prob > args.un_threshold:
            if args.verbose:
                print(f"\033[91mPhrase [{idx}]: {phrase}, P(</UN>): {prob:.4%}\033[0m")
            if back_track_idx == -1:
                if idx == 0:
                    latest_idx = NP_probs[idx][-1] + 1
                    if skip_first:
                        hallucination_phrase = phrase
                        continue
                    back_track_idx = 0
                else:
                    back_track_idx = NP_probs[idx - 1][-1] + 1 # backtrack to the end of last confident NP
                hallucination_phrase = phrase
        else:
            if args.verbose:
                print(f"Phrase [{idx}]: {phrase}, P(</UN>): {prob:.4%}")
    # 5. Return the results
    return readable_output, raw_output, new_output_ids, hallucination_phrase, back_track_idx, latest_idx


def back_track_last_sentence(existing_output_ids, tokenizer):
    """
    Backtrack to the previous <SPAN> token to avoid hallucination.
    """
    last_period_id = tokenizer.convert_tokens_to_ids(".")
    last_period_indices = torch.where(existing_output_ids == last_period_id)[1]
    if len(last_period_indices) == 0:
        truncated_content = tokenizer.decode(existing_output_ids[0], skip_special_tokens=True)
        truncated_content = truncated_content.replace("</UN>", "").replace("</CN>", "").replace("<SPAN>", "").strip()
        return existing_output_ids[:, :0], truncated_content
    last_period_indice = last_period_indices[-1]
    truncated_part = existing_output_ids[:, last_period_indice + 1:]
    # filter out the special tokens
    truncated_content = tokenizer.decode(truncated_part[0], skip_special_tokens=True)
    truncated_content = truncated_content.replace("</UN>", "").replace("</CN>", "").replace("<SPAN>", "").strip()
    return existing_output_ids[:, :last_period_indice + 1], truncated_content



@torch.inference_mode()
def do_inference(model, tokenizer, current_prefix, image_tensor, image_sizes, args, temperature):
    inputs = {
        "input_ids": current_prefix,
        "attention_mask": current_prefix.ne(tokenizer.pad_token_id),
        "pixel_values": image_tensor,
        "image_grid_thw": image_sizes,
    }
    output = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        return_dict_in_generate=True, 
        output_logits=True
    )
    output.sequences = output.sequences[:, len(current_prefix[0]):]
    # assert output.sequences.shape[1] == output.logits.shape[1], "Output sequence length does not match logits length."
    return output


def early_exit(existing_output_ids, tokenizer, generation_history, hallucination_phrase, cut_current_sentence=True):
    span_token_id = tokenizer.convert_tokens_to_ids("<SPAN>")
    span_indices = torch.where(existing_output_ids == span_token_id)[1]
    un_token_id = tokenizer.convert_tokens_to_ids("</UN>")
    cn_token_id = tokenizer.convert_tokens_to_ids("</CN>")
    un_indices = torch.where(existing_output_ids == un_token_id)[1]
    cn_indices = torch.where(existing_output_ids == cn_token_id)[1]
    raw_text = tokenizer.batch_decode(existing_output_ids, skip_special_tokens=True)[0].strip()

    special_indices = torch.cat([span_indices, un_indices, cn_indices], dim=0)
    readable_output = render_output(existing_output_ids[0], tokenizer, special_indices, cut_current_sentence=cut_current_sentence)

    generation_history.append((raw_text, hallucination_phrase))
    successful_correction = False
    return readable_output, generation_history, successful_correction



def retrospective_resampling_generate(
    model, tokenizer,
    inputs, args, add_open_ended_qa_prompt=False,
) -> tuple[str, list, bool]:
    """
    Generate text using rejection sampling to handle hallucinations.
    Args:
        model: The VLM model
        tokenizer: The tokenizer
        original_input_ids: Input token ids
        image_tensor: Image tensor input
        image_sizes: List of image sizes
        args: Generation arguments
        add_open_ended_qa_prompt: Add open-ended QA prompt if True
    Returns:
        tuple of (decoded_text, generation_history, successful_correction)
    """

    # Initialize generation parameters
    max_total_attempts = getattr(args, "max_total_generation_attempts", 50)
    max_local_attempts = getattr(args, "max_local_generation_attempts", 10)
    temperature_step = getattr(args, "temperature_step", 0.05)

    original_input_ids = inputs.input_ids
    image_tensor = inputs.pixel_values
    image_sizes = inputs.image_grid_thw
    instruction_length = original_input_ids.shape[1]

    # Initialize generation state
    existing_output_ids = original_input_ids[:, instruction_length:]
    generation_history = []
    all_hallucination_phrases = []
    successful_correction = False
    correction_mode = False
    local_attempt = 0
    temperature = args.temperature

    def prepare_prefix():
        if len(all_hallucination_phrases) == 0 and (not add_open_ended_qa_prompt):
            return torch.cat([original_input_ids, existing_output_ids], dim=1)
            
        # Add hint for query rewriting
        hint_string = " " + OPEN_ENDED_QA_QUERY_REWRITING_PROMPT  if add_open_ended_qa_prompt else " "
        if len(all_hallucination_phrases) > 0:
            negative_object_string = GENERAL_QUERY_REWRITING_PROMPT.format(", ".join(all_hallucination_phrases[-5:]))
            negative_object_string = " " + negative_object_string
            hint_string += negative_object_string
        hint_ids = tokenizer.encode(hint_string, return_tensors="pt").to(original_input_ids.device)[:, 1:]
        
        # Find assistant token position
        # special case for both LLaVA-v1.5-7B and LLaVA-MORE (llama-3.1-8B)
        # Assistant tokens are the last 5 tokens
        assistant_idx = -5 
        return torch.cat([
            original_input_ids[:, :assistant_idx],
            hint_ids,
            original_input_ids[:, assistant_idx:],
            existing_output_ids
        ], dim=1)

    def handle_empty_output(output_ids):
        span_token_id = tokenizer.convert_tokens_to_ids("<SPAN>")
        first_span = torch.where(output_ids.squeeze(0) == span_token_id)[0]
        if len(first_span) == 0:
            return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        back_track_idx = first_span[0]
        return tokenizer.decode(output_ids[0, :back_track_idx], skip_special_tokens=True).strip().lower()

    # Main generation loop
    for global_attempt in range(max_total_attempts):
        if not correction_mode:
            temperature = args.temperature
            
        if args.verbose:
            print("-" * 50)
            print(f"Rejection sampling attempt: {global_attempt}({local_attempt}) | Temperature: {temperature:.2f}")

        # Prepare input and generate
        current_prefix = prepare_prefix()
        # print(f"Current prefix: {tokenizer.decode(current_prefix[0], skip_special_tokens=True)}")
        # exit()
        if args.verbose:
            visualize_prefix = current_prefix.clone()
            print("Prefix: ", tokenizer.decode(visualize_prefix[0]))
        output = do_inference(model, tokenizer, current_prefix, image_tensor, image_sizes, args, temperature)
        output_ids, scores = output.sequences, output.logits
        
        # Check for hallucinations
        decoded_text, raw_text, output_ids, hallucination_phrase, back_track_idx, latest_idx = \
            de_hallucination_postprocessing(existing_output_ids, output_ids, scores, tokenizer, args, correction_mode=correction_mode)

        # Handle successful generation
        if hallucination_phrase is None:
            if not correction_mode:
                generation_history.append((raw_text, hallucination_phrase))
                successful_correction = True
                assert back_track_idx == -1, "Backtrack index should be -1 if no hallucination is detected."
                return decoded_text, generation_history, successful_correction
                
            # Handle successful correction
            correction_mode = False
            generation_history.append((raw_text, "successful correction"))
            existing_output_ids = torch.cat([existing_output_ids, output_ids[0, :back_track_idx].unsqueeze(0)], dim=1)
            local_attempt = 0
            continue

        # Handle hallucination
        assert back_track_idx != -1, "Backtrack index should not be -1 if hallucination is detected."
        correction_mode = True
        local_attempt += 1
        temperature = min(temperature + temperature_step, args.temperature + 0.5)
        
        if hallucination_phrase.lower() not in all_hallucination_phrases:
            all_hallucination_phrases.append(hallucination_phrase.lower())

        # Handle max local attempts reached
        if local_attempt == max_local_attempts:
            if len(existing_output_ids[0]) == 0:
                truncated_content = handle_empty_output(output_ids)
            else:
                existing_output_ids, truncated_content = back_track_last_sentence(existing_output_ids, tokenizer)

            if truncated_content:
                all_hallucination_phrases.append(truncated_content)

            raw_text = tokenizer.batch_decode(existing_output_ids, skip_special_tokens=True)[0].strip()
            generation_history.append((raw_text, hallucination_phrase))

            if args.verbose:
                print("Max local rejection attempts reached. Backtracking to the last sentence.")
                print(f"Curr output: {raw_text}")

            local_attempt = 0
            continue

        # Update generation state
        raw_text = tokenizer.batch_decode(
            torch.cat([existing_output_ids, output_ids[0, :latest_idx].unsqueeze(0)], dim=1),
            skip_special_tokens=True
        )[0].strip()
        generation_history.append((raw_text, hallucination_phrase))
        
        if args.verbose:
            print(f"Curr output: {raw_text}")
            
        existing_output_ids = torch.cat([existing_output_ids, output_ids[0, :back_track_idx].unsqueeze(0)], dim=1)

    if args.verbose:
        print(f"Max rejection attempts reached. Returning the last generated output: {decoded_text}")

    return early_exit(existing_output_ids, tokenizer, generation_history, hallucination_phrase, cut_current_sentence=args.open_ended_qa)