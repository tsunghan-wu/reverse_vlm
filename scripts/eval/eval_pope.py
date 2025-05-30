"""
Evaluation script for the POPE dataset.
Source: The code is adapted from LLaVA's evaluation script 
https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/eval_pope.py
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

def eval_pope(answers, label_file, n_bootstrap=1000, seed=42):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']
        if text.find('.') != -1:
            text = text.split('.')[0]
        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    label_bin = [0 if l == 'no' else 1 for l in label_list]
    pred_bin = [0 if ans['text'] == 'no' else 1 for ans in answers]
    assert len(label_bin) == len(pred_bin)

    def compute_metrics(preds, labels):
        TP = sum([p == 1 and l == 1 for p, l in zip(preds, labels)])
        FP = sum([p == 1 and l == 0 for p, l in zip(preds, labels)])
        TN = sum([p == 0 and l == 0 for p, l in zip(preds, labels)])
        FN = sum([p == 0 and l == 1 for p, l in zip(preds, labels)])

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (TP + TN) / (TP + TN + FP + FN)
        yes_ratio = sum(preds) / len(preds)

        return dict(f1=f1, acc=acc, precision=precision, recall=recall, yes_ratio=yes_ratio)

    # Bootstrap
    np.random.seed(seed)
    N = len(label_bin)
    metric_samples = defaultdict(list)

    for _ in range(n_bootstrap):
        indices = np.random.choice(N, N, replace=True)
        sample_preds = [pred_bin[i] for i in indices]
        sample_labels = [label_bin[i] for i in indices]
        metrics = compute_metrics(sample_preds, sample_labels)
        for k, v in metrics.items():
            metric_samples[k].append(v)

    print('Bootstrap Results (mean ± 95% CI):')
    for k in ['f1', 'acc', 'precision', 'recall', 'yes_ratio']:
        scores = np.array(metric_samples[k])
        mean = np.mean(scores)
        lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)
        print(f'{k}: {mean:.3f} ± ({lower:.3f}, {upper:.3f})')

    return metric_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    overall_metric_samples = defaultdict(list)
    for file in os.listdir(args.annotation_dir):
        if file.startswith('coco_pope_') and file.endswith('.json'):
            category = file[10:-5]
            cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
            print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
            metric_samples = eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
            for k, v in metric_samples.items():
                overall_metric_samples[k].extend(v)
            print("====================================")

    print("Final Average Across All Categories (Bootstrap 95% CI):")
    for k in ['f1', 'acc', 'precision', 'recall', 'yes_ratio']:
        scores = np.array(overall_metric_samples[k])
        mean = np.mean(scores)
        lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)
        print(f'{k:>10}: {mean:.3f} 95% CI: ({lower:.3f}-{upper:.3f})')
