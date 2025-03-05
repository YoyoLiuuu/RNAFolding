import os
import argparse
import torch
import torch.nn as nn

from model import *
from train import *
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Train RNA FM with LoRA')
    parser.add_argument('--data_path', type=str, help='Path to RNA sequences folder')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    args = parser.parse_args()

    model, alphabet = fm.pretrained.rna_fm_t12()
    model = add_lora(model)
    dataset = RNADataset(alphabet=alphabet, folder_path=None) # Add Folder_path
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate,
        shuffle=True
    )

    # Start training
    train(
        model,
        dataloader,
        checkpoint_dir=args.checkpoint_dir,
        
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    main()
