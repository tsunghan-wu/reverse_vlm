import os
import nltk
import json
import spacy
import warnings
import argparse
import numpy as np
from nltk.stem import WordNetLemmatizer

# Load language model and filter warnings
nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings("ignore", category=UserWarning)


########################################
#          Argument Parsing            #
########################################

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_dir", type=str, default='llava/eval/amber_data')
    parser.add_argument("--inference_data", type=str, required=True)
    parser.add_argument("--similarity_score", type=float, default=0.8)
    parser.add_argument('--evaluation_type', choices=['a', 'g', 'd', 'de', 'da', 'dr'],
                        help='a: all tasks and dimensions, g: generative task, d: discriminative task,'
                             ' de, da, dr: existence, attribute, relation')
    args = parser.parse_args()
    args.word_association = os.path.join(args.metadata_dir, 'relation.json')
    args.safe_words = os.path.join(args.metadata_dir, 'safe_words.txt')
    args.annotation = os.path.join(args.metadata_dir, 'annotations.json')
    args.metrics = os.path.join(args.metadata_dir, 'metrics.txt')
    return args


########################################
#          Utility Functions           #
########################################

def load_json(file_path):
    """Load JSON file and return the data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_text_lines(file_path):
    """Load text file and return list of stripped lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def load_inference_data(file_path):
    """Load inference data which is a JSON-lines file."""
    inference_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            inference_data.append(json.loads(line))
    return inference_data


def load_metrics(metrics_path):
    """Initialize and return a metrics dict based on a metrics file with key=value lines."""
    metrics = {}
    with open(metrics_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split('=')
        if len(parts) == 2:
            variable_name = parts[0].strip()
            variable_value = eval(parts[1].strip())
            metrics[variable_name] = variable_value
    return metrics


def check_synonyms_word(word1, word2, similarity_threshold):
    """Check if two words are synonyms based on spaCy similarity."""
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_threshold


def extract_nouns(text):
    """Extract lemmatized nouns from given text using NLTK."""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns


########################################
#       Evaluation Computations        #
########################################

def setup_dimensions(evaluation_type):
    """Setup which evaluation dimensions to run based on the evaluation_type argument."""
    dimensions = {'g': False, 'de': False, 'da': False, 'dr': False}
    if evaluation_type == 'a':
        for key in dimensions:
            dimensions[key] = True
    elif evaluation_type == 'g':
        dimensions['g'] = True
    elif evaluation_type == 'd':
        # Enable all discriminative tasks: existence, attribute, relation.
        dimensions['de'] = True
        dimensions['da'] = True
        dimensions['dr'] = True
    else:
        dimensions[evaluation_type] = True
    return dimensions


def prepare_association(association_file):
    """Load word associations and return a set of hallucination words (both keys and values)."""
    association = load_json(association_file)
    hallucination_words = set()
    for word1, related in association.items():
        hallucination_words.add(word1)
        hallucination_words.update(related)
    return association, hallucination_words


def process_generative_task(data_item, ground_truth_item, association, hallucination_words,
                            global_safe_words, similarity_threshold, metrics):
    """Process a generative task item and update the metrics dictionary accordingly."""
    question_id = data_item['question_id']
    # Extract nouns from text and keep only those in the association
    nouns = extract_nouns(data_item['text'])
    filtered_nouns = [noun for noun in nouns if noun in hallucination_words]

    # Build safe and hallucination lists based on association dictionary
    safe_words = []
    safe_list = []
    for idx, word in enumerate(ground_truth_item['truth']):
        # Extend safe list with associated words
        related = association.get(word, [])
        safe_words += related
        safe_list += [idx] * len(related)

    ha_words = []
    ha_list = []
    for idx, word in enumerate(ground_truth_item['hallu']):
        # Extend hallucination list with associated words
        related = association.get(word, [])
        ha_words += related
        ha_list += [idx] * len(related)

    # Append direct words to the lists
    safe_words += ground_truth_item['truth']
    safe_len = len(ground_truth_item['truth'])
    safe_list += [0] * safe_len

    ha_words += ground_truth_item['hallu']
    ha_len = len(ground_truth_item['hallu'])
    ha_list += [0] * ha_len

    safe_flag_list = [0] * len(filtered_nouns)

    # Process each filtered noun
    for idx, noun in enumerate(filtered_nouns):
        if noun in global_safe_words:
            continue

        # Check if noun is explicitly safe
        if noun in safe_words:
            for j in range(len(safe_words)):
                if noun == safe_words[j]:
                    # Adjust the corresponding flag; note: the code logic is preserved from the original
                    if j < (len(safe_list) - safe_len):
                        safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                    else:
                        safe_list[j] = 1
                    break
            continue

        # Check if noun is explicitly hallucinated
        if noun in ha_words:
            for j in range(len(ha_words)):
                if noun == ha_words[j]:
                    if j < (len(ha_list) - ha_len):
                        ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                    else:
                        ha_list[j] = 1
                    break

        # Check synonyms in hallucination words
        for j, check_word in enumerate(ha_words):
            if check_synonyms_word(noun, check_word, similarity_threshold):
                if j < (len(ha_list) - ha_len):
                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                else:
                    ha_list[j] = 1
                break

        # Check synonyms in safe words
        flag = False
        for j, check_word in enumerate(safe_words):
            if check_synonyms_word(noun, check_word, similarity_threshold):
                flag = True
                if j < (len(safe_list) - safe_len):
                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                else:
                    safe_list[j] = 1
                break
        if flag:
            continue

        safe_flag_list[idx] = 1

    # Update metrics for generative tasks
    metrics['chair_score'] += sum(safe_flag_list)
    metrics['chair_num'] += len(safe_flag_list)
    metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
    metrics['safe_cover_num'] += len(safe_list[-safe_len:])
    metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
    metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
    if sum(safe_flag_list) == 0:
        metrics['non_hallu_score'] += 1
    metrics['non_hallu_num'] += 1


def process_discriminative_task(data_item, ground_truth_item, metrics):
    """Process a discriminative task item and update the metrics dictionary accordingly."""
    question_id = data_item['question_id']
    metrics['qa_correct_num'] += 1

    # Update QA type based metrics (the ground truth type is expected to be one of the discriminative types)
    gt_type = ground_truth_item['type']
    if gt_type == 'discriminative-attribute-state':
        metrics['as_qa_correct_num'] += 1
    elif gt_type == 'discriminative-attribute-number':
        metrics['an_qa_correct_num'] += 1
    elif gt_type == 'discriminative-attribute-action':
        metrics['aa_qa_correct_num'] += 1
    elif gt_type == 'discriminative-hallucination':
        metrics['ha_qa_correct_num'] += 1
    else:
        metrics['asso_qa_correct_num'] += 1

    truth = ground_truth_item['truth']
    # Standardize the response
    response = data_item['text']
    if "yes" in response.lower():
        response = "Yes"
    elif "no" in response.lower():
        response = "No"

    # Process the answer matching
    if truth == 'yes':
        if response == "Yes":
            metrics['qa_correct_score'] += 1
            if gt_type == 'discriminative-attribute-state':
                metrics['as_qa_correct_score'] += 1
            elif gt_type == 'discriminative-attribute-number':
                metrics['an_qa_correct_score'] += 1
            elif gt_type == 'discriminative-attribute-action':
                metrics['aa_qa_correct_score'] += 1
            elif gt_type == 'discriminative-hallucination':
                metrics['ha_qa_correct_score'] += 1
            else:
                metrics['asso_qa_correct_score'] += 1
    else:
        metrics['qa_no_num'] += 1
        if gt_type == 'discriminative-attribute-state':
            metrics['as_qa_no_num'] += 1
        elif gt_type == 'discriminative-attribute-number':
            metrics['an_qa_no_num'] += 1
        elif gt_type == 'discriminative-attribute-action':
            metrics['aa_qa_no_num'] += 1
        elif gt_type == 'discriminative-hallucination':
            metrics['ha_qa_no_num'] += 1
        else:
            metrics['asso_qa_no_num'] += 1

        if response == "No":
            metrics['qa_correct_score'] += 1
            metrics['qa_no_score'] += 1
            if gt_type == 'discriminative-attribute-state':
                metrics['as_qa_correct_score'] += 1
                metrics['as_qa_no_score'] += 1
            elif gt_type == 'discriminative-attribute-number':
                metrics['an_qa_correct_score'] += 1
                metrics['an_qa_no_score'] += 1
            elif gt_type == 'discriminative-attribute-action':
                metrics['aa_qa_correct_score'] += 1
                metrics['aa_qa_no_score'] += 1
            elif gt_type == 'discriminative-hallucination':
                metrics['ha_qa_correct_score'] += 1
                metrics['ha_qa_no_score'] += 1
            else:
                metrics['asso_qa_correct_score'] += 1
                metrics['asso_qa_no_score'] += 1

    if response == "No":
        metrics['qa_ans_no_num'] += 1
        if gt_type == 'discriminative-attribute-state':
            metrics['as_qa_ans_no_num'] += 1
        elif gt_type == 'discriminative-attribute-number':
            metrics['an_qa_ans_no_num'] += 1
        elif gt_type == 'discriminative-attribute-action':
            metrics['aa_qa_ans_no_num'] += 1
        elif gt_type == 'discriminative-hallucination':
            metrics['ha_qa_ans_no_num'] += 1
        else:
            metrics['asso_qa_ans_no_num'] += 1
        if truth == 'no':
            metrics['qa_ans_no_score'] += 1
            if gt_type == 'discriminative-attribute-state':
                metrics['as_qa_ans_no_score'] += 1
            elif gt_type == 'discriminative-attribute-number':
                metrics['an_qa_ans_no_score'] += 1
            elif gt_type == 'discriminative-attribute-action':
                metrics['aa_qa_ans_no_score'] += 1
            elif gt_type == 'discriminative-hallucination':
                metrics['ha_qa_ans_no_score'] += 1
            else:
                metrics['asso_qa_ans_no_score'] += 1


def compute_and_print_metrics(metrics, dimensions):
    """Compute percentages for each evaluation dimension and print them."""
    if dimensions['g']:
        CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1)
        Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1)
        Ha = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1)
        Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1)
        print("Generative Task:")
        print("CHAIR:\t\t", CHAIR)
        print("Cover:\t\t", Cover)
        print("Hal:\t\t", Ha_p)
        print("Cog:\t\t", Ha, "\n")

    if dimensions['de'] and dimensions['da'] and dimensions['dr']:
        Accuracy = round(metrics['qa_correct_score'] / metrics['qa_correct_num'] * 100, 1)
        Precision = round(metrics['qa_ans_no_score'] / metrics['qa_ans_no_num'] * 100, 1)
        Recall = round(metrics['qa_no_score'] / metrics['qa_no_num'] * 100, 1)
        F1 = round(2 * (Precision/100) * (Recall/100) / ((Precision/100) + (Recall/100) + 0.0001) * 100, 1)
        print("Discriminative Task:")
        print("Accuracy:\t", Accuracy)
        print("Precision:\t", Precision)
        print("Recall:\t\t", Recall)
        print("F1:\t\t", F1, "\n")

    if dimensions['de']:
        hallucination_Accuracy = round(metrics['ha_qa_correct_score'] / metrics['ha_qa_correct_num'] * 100, 1)
        hallucination_Precision = round(metrics['ha_qa_ans_no_score'] / metrics['ha_qa_ans_no_num'] * 100, 1)
        hallucination_Recall = round(metrics['ha_qa_no_score'] / metrics['ha_qa_no_num'] * 100, 1)
        hallucination_F1 = round(2 * (hallucination_Precision/100) * (hallucination_Recall/100) / ((hallucination_Precision/100) + (hallucination_Recall/100) + 0.001) * 100, 1)
        print("Existence:")
        print("Accuracy:\t", hallucination_Accuracy)
        print("Precision:\t", hallucination_Precision)
        print("Recall:\t\t", hallucination_Recall)
        print("F1:\t\t", hallucination_F1, "\n")

    if dimensions['da']:
        attr_Accuracy = round((metrics['as_qa_correct_score'] + metrics['an_qa_correct_score'] + metrics['aa_qa_correct_score']) / 
                                (metrics['as_qa_correct_num'] + metrics['an_qa_correct_num'] + metrics['aa_qa_correct_num']) * 100, 1)
        attr_Precision = round((metrics['as_qa_ans_no_score'] + metrics['an_qa_ans_no_score'] + metrics['aa_qa_ans_no_score']) / 
                                 (metrics['as_qa_ans_no_num'] + metrics['an_qa_ans_no_num'] + metrics['aa_qa_ans_no_num']) * 100, 1)
        attr_Recall = round((metrics['as_qa_no_score'] + metrics['an_qa_no_score'] + metrics['aa_qa_no_score']) / 
                              (metrics['as_qa_no_num'] + metrics['an_qa_no_num'] + metrics['aa_qa_no_num']) * 100, 1)
        attr_F1 = round(2 * (attr_Precision/100) * (attr_Recall/100) / ((attr_Precision/100) + (attr_Recall/100) + 0.0001) * 100, 1)
        state_Accuracy = round(metrics['as_qa_correct_score'] / metrics['as_qa_correct_num'] * 100, 1)
        state_Precision = round(metrics['as_qa_ans_no_score'] / metrics['as_qa_ans_no_num'] * 100, 1)
        state_Recall = round(metrics['as_qa_no_score'] / metrics['as_qa_no_num'] * 100, 1)
        state_F1 = round(2 * (state_Precision/100) * (state_Recall/100) / ((state_Precision/100) + (state_Recall/100) + 0.0001) * 100, 1)
        number_Accuracy = round(metrics['an_qa_correct_score'] / metrics['an_qa_correct_num'] * 100, 1)
        number_Precision = round(metrics['an_qa_ans_no_score'] / metrics['an_qa_ans_no_num'] * 100, 1)
        number_Recall = round(metrics['an_qa_no_score'] / metrics['an_qa_no_num'] * 100, 1)
        number_F1 = round(2 * (number_Precision/100) * (number_Recall/100) / ((number_Precision/100) + (number_Recall/100) + 0.0001) * 100, 1)
        action_Accuracy = round(metrics['aa_qa_correct_score'] / metrics['aa_qa_correct_num'] * 100, 1)
        action_Precision = round(metrics['aa_qa_ans_no_score'] / metrics['aa_qa_ans_no_num'] * 100, 1)
        action_Recall = round(metrics['aa_qa_no_score'] / metrics['aa_qa_no_num'] * 100, 1)
        action_F1 = round(2 * (action_Precision/100) * (action_Recall/100) / ((action_Precision/100) + (action_Recall/100) + 0.0001) * 100, 1)
        print("Attribute:")
        print("Accuracy:\t", attr_Accuracy)
        print("Precision:\t", attr_Precision)
        print("Recall:\t\t", attr_Recall)
        print("F1:\t\t", attr_F1, "\n")
        print("State:")
        print("Accuracy:\t", state_Accuracy)
        print("Precision:\t", state_Precision)
        print("Recall:\t\t", state_Recall)
        print("F1:\t\t", state_F1, "\n")
        print("Number:")
        print("Accuracy:\t", number_Accuracy)
        print("Precision:\t", number_Precision)
        print("Recall:\t\t", number_Recall)
        print("F1:\t\t", number_F1, "\n")
        print("Action:")
        print("Accuracy:\t", action_Accuracy)
        print("Precision:\t", action_Precision)
        print("Recall:\t\t", action_Recall)
        print("F1:\t\t", action_F1, "\n")

    if dimensions['dr']:
        relation_Accuracy = round(metrics['asso_qa_correct_score'] / metrics['asso_qa_correct_num'] * 100, 1)
        relation_Precision = round(metrics['asso_qa_ans_no_score'] / metrics['asso_qa_ans_no_num'] * 100, 1)
        relation_Recall = round(metrics['asso_qa_no_score'] / metrics['asso_qa_no_num'] * 100, 1)
        relation_F1 = round(2 * (relation_Precision/100) * (relation_Recall/100) / ((relation_Precision/100) + (relation_Recall/100) + 0.0001) * 100, 1)
        print("Relation:")
        print("Accuracy:\t", relation_Accuracy)
        print("Precision:\t", relation_Precision)
        print("Recall:\t\t", relation_Recall)
        print("F1:\t\t", relation_F1)


########################################
#                Main                  #
########################################

def get_generative_deltas(data_item, ground_truth_item, association,
                          hallucination_words, global_safe_words, similarity_threshold):
    temp_metrics = {
        'chair_score': 0, 'chair_num': 0,
        'safe_cover_score': 0, 'safe_cover_num': 0,
        'hallu_cover_score': 0, 'hallu_cover_num': 0,
        'non_hallu_score': 0, 'non_hallu_num': 0
    }
    process_generative_task(data_item, ground_truth_item, association,
                            hallucination_words, global_safe_words,
                            similarity_threshold, temp_metrics)
    return temp_metrics

def bootstrap_evaluation(args, n_bootstrap=100, seed=42):
    # Load files and initialize data
    association, hallucination_words = prepare_association(args.word_association)
    global_safe_words = load_text_lines(args.safe_words)
    dimensions = setup_dimensions(args.evaluation_type)
    inference_data = load_inference_data(args.inference_data)
    ground_truth = load_json(args.annotation)
    rng = np.random.default_rng(seed)

    # Precompute generative outputs and cache
    generative_cache = {}
    for i, data_item in enumerate(inference_data):
        gt_item = ground_truth[data_item['question_id'] - 1]
        if gt_item['type'] == 'generative':
            generative_cache[i] = get_generative_deltas(
                data_item, gt_item, association,
                hallucination_words, global_safe_words, args.similarity_score
            )

    all_scores = {
        'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [],
        'CHAIR': [], 'Cover': [], 'Hal': [], 'Cog': []
    }

    for _ in range(n_bootstrap):
        sample_indices = rng.choice(len(inference_data), len(inference_data), replace=True)
        metrics = load_metrics(args.metrics)

        for idx in sample_indices:
            data_item = inference_data[idx]
            gt_item = ground_truth[data_item['question_id'] - 1]

            if gt_item['type'] == 'generative':
                delta = generative_cache[idx]
                for k in delta:
                    metrics[k] += delta[k]
            else:
                process_discriminative_task(data_item, gt_item, metrics)

        # Compute metrics from this bootstrap sample
        if dimensions['g'] and metrics['chair_num'] > 0:
            all_scores['CHAIR'].append(metrics['chair_score'] / metrics['chair_num'] * 100)
            all_scores['Cover'].append(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100)
            all_scores['Cog'].append(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100)
            all_scores['Hal'].append(100 - (metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100))

        if dimensions['de'] and metrics['qa_correct_num'] > 0 and metrics['qa_ans_no_num'] > 0 and metrics['qa_no_num'] > 0:
            acc = metrics['qa_correct_score'] / metrics['qa_correct_num']
            prec = metrics['qa_ans_no_score'] / metrics['qa_ans_no_num']
            rec = metrics['qa_no_score'] / metrics['qa_no_num']
            f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
            all_scores['Accuracy'].append(acc * 100)
            all_scores['Precision'].append(prec * 100)
            all_scores['Recall'].append(rec * 100)
            all_scores['F1'].append(f1 * 100)

    print("Bootstrapped Evaluation (95% CI):")
    for key in ['Accuracy', 'Precision', 'Recall', 'F1']:
        if all_scores[key]:
            vals = np.array(all_scores[key])
            mean = np.mean(vals)
            lb, ub = np.percentile(vals, [2.5, 97.5])
            print(f"{key}:\t{mean:.1f} ({lb:.1f} – {ub:.1f})")

    for key in ['CHAIR', 'Cover', 'Hal', 'Cog']:
        if all_scores[key]:
            vals = np.array(all_scores[key])
            mean = np.mean(vals)
            lb, ub = np.percentile(vals, [2.5, 97.5])
            print(f"{key}:\t{mean:.1f} ({lb:.1f} – {ub:.1f})")


if __name__ == "__main__":
    args = get_args()
    bootstrap_evaluation(args, n_bootstrap=100)
