"""
Evaluation script for the HaloQuest dataset.
Source: The code is adapted from the official HaloQuest Colab notebook.
https://github.com/google/haloquest/blob/main/HaloQuest_Colab.ipynb
"""

import os
import json
import numpy as np
import langfun as lf
import pandas as pd
import pyglove as pg
import argparse
from tqdm import tqdm

api_key = os.environ.get("GOOGLE_API_KEY")

if api_key is None:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")


lm_model = lf.llms.GeminiPro1_5(api_key=api_key)


class PredictionEvaluation(pg.Object):
  question: str
  model_response: str
  groundtruth_response: str
  sentence_interpret_model_response_main_point: str
  sentence_interpret_groundtruth_response_main_point: str
  is_prediction_correct: bool

  def to_dict(self):
    return {
      "question": self.question,
      "model_response": self.model_response,
      "groundtruth_response": self.groundtruth_response,
      "sentence_interpret_model_response_main_point": self.sentence_interpret_model_response_main_point,
      "sentence_interpret_groundtruth_response_main_point": self.sentence_interpret_groundtruth_response_main_point,
      "is_prediction_correct": self.is_prediction_correct
    }


def compute_prediction(inputs):
  _, question, model_response, groundtruth_response = inputs

  r = lf.query(prompt="""
  Your task is to determine if the model response is correct given the question and groundtruth response.
  Ensure to interpret the model response in accordance to the the question.

  If the question asks about a detail of an element that is not present in the image, A prediction of "yes", "no" or "nothing" should be considered incorrect because it inaccurately suggests that the element is presented in the image.
  The correct prediction in such cases should acknowledge the absence of the element in question by stating the element is not present.
  If prediction says that it can not assist or cannot provide an answer, then the prediction is incorrect.
  If the question is about counting, then the prediction is correct only it matches the groundtruth counts exactly.

  question={{question}},
  model_response={{model_response}}
  groundtruth_response={{groundtruth_response}},

  """,
  schema=PredictionEvaluation,
  question=question,
  groundtruth_response=groundtruth_response,
  model_response=model_response,
  lm=lm_model,
  autofix_lm=lm_model)

  return r.is_prediction_correct, r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--evaluation-result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {prediction['question_id']: prediction for prediction in predictions}

    eval_results = []
    for question_id, question in tqdm(questions.items()):
        prediction = predictions[question_id]
        answer = question['answer']
        try:
            result = compute_prediction(inputs=(question_id, question['text'], prediction, answer))
            correctness, log = result

            log = log.to_dict()
            log["hallucination_type"] = question["hallucination_type"]
            log["image_type"] = question["image_type"]
            log["question_id"] = question_id
            eval_results.append(log)
        except:
           print(f"Failed to evaluate question {question_id}")

    with open(args.evaluation_result_file, "w") as f:
        for log in eval_results:
            f.write(json.dumps(log) + "\n")

    # Convert eval_results to DataFrame
    df = pd.DataFrame(eval_results)
    df["score"] = df["is_prediction_correct"].astype(float)

    # Bootstrap configuration
    n_bootstrap = 100
    n_samples = len(df)
    bootstrap_avg_scores = []
    bootstrap_avg_scores_per_type = {ht: [] for ht in df["hallucination_type"].unique()}

    for _ in range(n_bootstrap):
        sampled_df = df.sample(n=n_samples, replace=True)

        # Overall
        bootstrap_avg_scores.append(sampled_df["score"].mean())
        # Per hallucination_type
        for ht, group in sampled_df.groupby("hallucination_type"):
            bootstrap_avg_scores_per_type[ht].append(group["score"].mean())

    # Report confidence intervals
    def ci(data):
        return np.percentile(data, [2.5, 97.5])

    print("Average score: {:.3f}, 95% CI: {:.3f}-{:.3f}".format(
        df["score"].mean(), *ci(bootstrap_avg_scores)))

    print("\nAverage score per hallucination type:")
    for ht in sorted(bootstrap_avg_scores_per_type):
        mean_score = df[df["hallucination_type"] == ht]["score"].mean()
        lb, ub = ci(bootstrap_avg_scores_per_type[ht])
        print(f"  {ht}: {mean_score:.3f}, 95% CI: {lb:.3f}-{ub:.3f}")
