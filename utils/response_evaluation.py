"""Script evaluates response generation using GT responses.

Expected JSON format:

[
    "dialog_id": <dialog_id>,
    "predictions": [
        {
            "turn_id": <turn_id>,
            "response": <str; model output>,
        }
        ...
    ]
    ...
]

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json

import nltk
import numpy as np


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize.
    """
    return nltk.tokenize.word_tokenize(sentence.lower())


def evaluate_response_generation(
    gt_responses, model_responses, single_round_eval=False
):
    """Evaluates response generation using the raw data and model predictions.

    Args:
        gt_responses: Ground truth responses.
        model_responses: Generated responses.
        single_round_eval: Evaluate only for the last turn.
    """
    gt_responses_pool = {ii["dialogue_idx"]: ii for ii in gt_responses["dialogue_data"]}
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    num_evaluations = 0
    for model_datum in model_responses:
        dialog_id = model_datum["dialog_id"]
        num_gt_rounds = len(gt_responses_pool[dialog_id]["dialogue"])
        for round_datum in model_datum["predictions"]:
            round_id = round_datum["turn_id"]
            # Skip if single_round_eval and this is not the last round.
            if single_round_eval and round_id != num_gt_rounds - 1:
                continue

            response = round_datum["response"]
            gt_datum = gt_responses_pool[dialog_id]["dialogue"][round_id]
            gt_response = gt_datum["system_transcript"]

            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(gt_response)],
                normalize_sentence(response),
                smoothing_function=chencherry.method7,
            )
            bleu_scores.append(bleu_score)
    print("#Instances evaluated BLEU: {}".format(len(bleu_scores)))
    return np.mean(bleu_scores), np.std(bleu_scores) / np.sqrt(len(bleu_scores))


def main(args):
    print("Reading: {}".format(args["data_json_path"]))
    with open(args["data_json_path"], "r") as file_id:
        gt_responses = json.load(file_id)
    print("Reading: {}".format(args["model_response_path"]))
    with open(args["model_response_path"], "r") as file_id:
        model_responses = json.load(file_id)
    bleu_score, bleu_std_err = evaluate_response_generation(
        gt_responses, model_responses, args["single_round_evaluation"]
    )
    print(f"BLEU Score: {bleu_score:.4f} +- {bleu_std_err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Response Generation Evaluation")
    parser.add_argument(
        "--data_json_path",
        default="data/furniture_train.json",
        help="Data with gold responses",
    )
    parser.add_argument(
        "--model_response_path", default=None, help="Responses generated by the model"
    )
    parser.add_argument(
        "--single_round_evaluation",
        dest="single_round_evaluation",
        action="store_true",
        default=False,
        help="Single round evaluation for hidden split",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
