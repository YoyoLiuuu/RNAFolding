import argparse
import torch
import numpy as np
import pandas as pd
from model import finetuned_RibonanzaNet, load_config_from_yaml
from hungarian import mask_diagonal
from arnie.pk_predictors import _hungarian
from dataset import RNA2D_set

# load model
CONFIG_PATH = "ribonanzanet2d/configs/pairwise.yaml"
MODEL_PATH = "weight/RibonanzaNet-SS.pt"

# initialize model
model = finetuned_RibonanzaNet(load_config_from_yaml(CONFIG_PATH)).cuda()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def predict_structure(sequence):
 
    # take single sequence as input
    test_dataset = RNA2D_set([{"sequence": sequence}])

    # run inference
    with torch.no_grad():
        pred_matrix = model(test_dataset[0]["sequence"].cuda().unsqueeze(0)).sigmoid().cpu().numpy()

    # apply hungarian get for secondary structure
    s, bp = _hungarian(mask_diagonal(pred_matrix[0]), theta=0.5, min_len_helix=1)

    return s


def main():
    parser = argparse.ArgumentParser(description="Predict RNA Secondary Structure")
    parser.add_argument("sequence", type=str, help="RNA sequence (e.g., AUGCUAGCUAGC)")
    args = parser.parse_args()

    # input sequence must be AUGC
    valid_nucleotides = {"A", "U", "G", "C"}
    if not set(args.sequence.upper()).issubset(valid_nucleotides):
        print("Error: Invalid RNA sequence. Only A, U, G, and C are allowed.")
        return

    # predict
    structure = predict_structure(args.sequence.upper())

    # result
    print(f"Input Sequence: {args.sequence}")
    print(f"Predicted Structure: {structure}")


if __name__ == "__main__":
    main()