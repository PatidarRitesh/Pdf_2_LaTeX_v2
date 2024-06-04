
import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import logging
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pdf_2_tex import PDF_2_TEX_Model
from pdf_2_tex.metrics import compute_metrics
from pdf_2_tex.utils.checkpoint import get_checkpoint
from pdf_2_tex import pdf_2_tex_Dataset,test_dataset
from pdf_2_tex.utils.device import move_to_device
from lightning_module import PDF_2_TEX_DataPLModule


def test(args):
    pretrained_model = PDF_2_TEX_Model.from_pretrained(args.checkpoint)
    pretrained_model = move_to_device(pretrained_model)

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    else:
        logging.warning("Results can not be saved. Please provide a -o/--save_path")
    predictions = []
    ground_truths = []
    metrics = defaultdict(list)
    dataset = test_dataset(
        dataset_path=args.dataset,
        pdf_2_tex_model=pretrained_model,
        max_length=pretrained_model.config.max_length,
        split=args.split,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=args.shuffle,
        collate_fn=PDF_2_TEX_DataPLModule.ignore_none_collate,
    )

    for idx, sample in tqdm(enumerate(dataloader), total=1000):
        if sample is None:
            continue
        image_tensors, txt_input_tensor, decoder_input_ids, _ ,latex_path= sample
        if image_tensors is None:
            return
        if len(predictions) >= args.num_samples:
            break
        ground_truth = pretrained_model.decoder.tokenizer.batch_decode(
            decoder_input_ids, skip_special_tokens=True
        )
        outputs = pretrained_model.inference(
            image_tensors=image_tensors,
            text_tensors=txt_input_tensor,
            return_attentions=False,
        )["predictions"]
        predictions.extend(outputs)
        ground_truths.extend(ground_truth)
        with Pool(args.batch_size) as p:
            _metrics = p.starmap(compute_metrics, iterable=zip(outputs, ground_truth))
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
    except:
        pass
    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        with open(args.save_path, "w") as f:
            json.dump(scores, f)
    
    
    score_path='/mnt/NAS/patidarritesh/Pdf_2_LaTeX_v2_LONGFORMER/Scores.json'
    with open(score_path, 'w') as f:
        json.dump(only_scores, f)
    
    # if not os.path.exists('./Outputs'):
    #     os.makedirs('./Outputs')

    # # Ensure latex_path is a string if it's a list (similar handling as before)
    # if isinstance(latex_path, list):
    #     latex_path = '_'.join(latex_path)

    # file_path = os.path.join('./Outputs', latex_path)

    # # Process predictions assuming it's the source of latex_string
    # if isinstance(predictions, list):
    #     # Join the list into a single string with new lines
    #     latex_string = '\n'.join(predictions)

    # # Replace unwanted substrings
    # formatted_latex = latex_string.replace('[PAD]', '').replace('\\n', '\n')

    # with open(file_path, 'w', encoding='utf-8') as tex_file:
    #     tex_file.write(formatted_latex)

    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--save_path", "-o", type=str, default=None, help="json file to save results to"
    )
    parser.add_argument("--num_samples", "-N", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    args, left_argv = parser.parse_known_args()
    args.checkpoint = get_checkpoint(args.checkpoint)

    predictions = test(args)