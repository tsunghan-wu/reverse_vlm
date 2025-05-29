"""
Evaluation script for the MME dataset.
Source: The code is adapted from LLaVA's evaluation script
https://github.com/haotian-liu/LLaVA
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score

eval_type_dict = {
    # "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    # "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"],
    "Hallucination": ["existence", "count", "position", "color"]
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    return parser.parse_args()


def convert_answers(pred_file, results_dir):
    GT = {}
    data = load_dataset("lmms-lab/MME", data_dir="data")["test"]
    for entry in data:
        category = entry['category']
        file = entry['question_id'].split('/')[-1].split('.')[0] + '.txt'
        question = entry['question']
        GT[(category, file, question)] = entry['answer']

    os.makedirs(results_dir, exist_ok=True)
    answers = [json.loads(line) for line in open(pred_file)]
    results = defaultdict(list)

    for answer in answers:
        category = answer['question_id'].split('/')[0]
        file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
        question = answer['prompt']
        if 'Answer the question using a single word or phrase.' in question:
            question = question.replace('Answer the question using a single word or phrase.', '').strip()
        if 'Please answer yes or no.' not in question:
            question += ' Please answer yes or no.'
            if (category, file, question) not in GT:
                question = question.replace(' Please answer yes or no.', '  Please answer yes or no.')
        gt_ans = GT[(category, file, question)]
        results[category].append((file, question, gt_ans, answer['text']))

    for category, items in results.items():
        with open(os.path.join(results_dir, f'{category}.txt'), 'w') as fp:
            for file, question, gt_ans, pred_ans in items:
                fp.write('\t'.join([file, question, gt_ans, pred_ans]) + '\n')


class MME_Evaluator:

    def divide_chunks(self, l, n=2):
        for i in range(0, len(l), n): 
            yield l[i:i + n]

    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower()
        if pred_ans in ["yes", "no"]:
            return pred_ans
        if "yes" in pred_ans[:4]:
            return "yes"
        elif "no" in pred_ans[:4]:
            return "no"
        return "other"

    def compute_metric(self, gts, preds):
        label_map = {"yes": 1, "no": 0, "other": -1}
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts, clean_preds = [], []
        for gt, pred in zip(gts, preds):
            if pred == -1:
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        precision = precision_score(clean_gts, clean_preds, zero_division=0)
        recall = recall_score(clean_gts, clean_preds, zero_division=0)
        return acc, precision, recall

    def bootstrap_score(self, scores, n_rounds=100):
        arr = np.array(scores)
        n = len(arr)
        samples = [np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(n_rounds)]
        mean = np.mean(samples)
        low = np.percentile(samples, 2.5)
        high = np.percentile(samples, 97.5)
        return mean, low, high
    
    def bootstrap_total_score(self, all_task_scores, n_rounds=100):
        # all_task_scores: list of lists of per-image scores (one per task)
        total_samples = []
        for _ in range(n_rounds):
            task_means = [
                np.mean(np.random.choice(task_scores, size=len(task_scores), replace=True))
                for task_scores in all_task_scores
            ]
            total_samples.append(np.sum(task_means))
        mean = np.mean(total_samples)
        low = np.percentile(total_samples, 2.5)
        high = np.percentile(total_samples, 97.5)
        return mean, low, high

    def process_result(self, results_dir):
        
        for eval_type, task_name_list in eval_type_dict.items():
            print(f"\n=========== {eval_type} ===========\n")
            total_score = 0
            all_task_scores = []
            for task_name in task_name_list:
                task_txt = os.path.join(results_dir, task_name + ".txt")
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines))

                score_per_img = []
                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    correct = 0
                    gts, preds = [], []

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.strip().split("\t")
                        gt_ans = gt_ans.lower()
                        pred_ans = self.parse_pred_ans(pred_ans)

                        assert gt_ans in ["yes", "no"]
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)

                        if gt_ans == pred_ans:
                            correct += 1

                    acc = correct / 2
                    acc_plus = 1 if correct == 2 else 0
                    score = (acc + acc_plus) * 100
                    score_per_img.append(score)
                all_task_scores.append(score_per_img)
                mean_score, low, high = self.bootstrap_score(score_per_img)
                total_score += mean_score
                print(f"[{task_name}] score: {mean_score:.2f} (95% CI: {low:.2f}–{high:.2f})")
            total_mean, total_low, total_high = self.bootstrap_total_score(all_task_scores)
            print(f"\nTotal Score: {total_mean:.2f} (95% CI: {total_low:.2f}–{total_high:.2f})")

if __name__ == "__main__":
    args = get_args()
    convert_answers(args.pred_file, args.results_dir)
    cal = MME_Evaluator()
    cal.process_result(args.results_dir)
