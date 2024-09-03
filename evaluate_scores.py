import os
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from pdf_2_tex.metrics import compute_metrics

def evaluate_latex_predictions(ground_truth_dir, predicted_dir, save_path, batch_size=4, expno=0):
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    predicted_files = sorted(os.listdir(predicted_dir))

    predictions = []
    ground_truths = []
    metrics = {'edit_dist': [], 'bleu': [], 'meteor': [], 'precision': [], 'recall': [], 'f_measure': []}

    for idx, (gt_file, pred_file) in tqdm(enumerate(zip(ground_truth_files, predicted_files)), total=len(ground_truth_files)):
        
        with open(os.path.join(ground_truth_dir, gt_file), 'r') as gt_f:
            ground_truth = gt_f.read().strip()

        with open(os.path.join(predicted_dir, pred_file), 'r') as pred_f:
            prediction = pred_f.read().strip()

        predictions.append(prediction)
        ground_truths.append(ground_truth)

        with Pool(batch_size) as p:
            _metrics = p.starmap(compute_metrics, [(prediction, ground_truth)])
            for m in _metrics:
                for key, value in m.items():
                    metrics[key].append(value)

        print({key: sum(values) / len(values) for key, values in metrics.items()})

    scores = {}
    only_scores = {}

    for metric, vals in metrics.items():
        scores[f"{metric}_accuracies"] = vals
        scores[f"{metric}_accuracy"] = np.mean(vals)
        only_scores[f"{metric}_accuracies"] = vals
        only_scores[f"{metric}_accuracy"] = np.mean(vals)

    try:
        print(
            f"Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except KeyError:
        pass

    if save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths

        save_file = os.path.join(save_path, f"eval_scores_{expno}.json")
        
        with open(save_file, "w") as f:
            json.dump(scores, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LaTeX Predictions")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Path to the ground truth LaTeX files")
    parser.add_argument("--predicted_dir", type=str, required=True, help="Path to the predicted LaTeX files")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the evaluation scores")
    # parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for parallel processing")
    parser.add_argument("--expno", type=int, default=0, help="")

    args = parser.parse_args()

    evaluate_latex_predictions(
        ground_truth_dir=args.ground_truth_dir,
        predicted_dir=args.predicted_dir,
        save_path=args.save_path,
        # num_samples=args.num_samples,
        batch_size=args.batch_size,
        expno=args.expno
    )

# python evaluate_scores.py --ground_truth_dir /path/to/ground_truth --predicted_dir /path/to/predicted --save_path /path/to/save/scores.json