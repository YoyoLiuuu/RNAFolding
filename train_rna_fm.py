import os
import argparse
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import RNADataset, custom_collate
from src.lora import add_lora
import torchinfo

import tqdm

from src.utils import *
from src.model import load_rna_fm_t12, RNAFMLitModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train RNA FM with LoRA')
    parser.add_argument('--data_path', type=str, help='Path to RNA sequences folder')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    return args

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = load_rna_fm_t12()
    dataset = RNADataset(alphabet=alphabet, folder_path=args.data_path)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True)
    
    lit_model = RNAFMLitModel(model, learning_rate=args.lr)
    lit_model.to(device)  # Ensure the model is also on the same device

    logger = TensorBoardLogger("logs", name="rna_fm_lora")
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=1,
        mode="min",
        dirpath=args.checkpoint_dir,
        filename="best_model"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    
    trainer.fit(lit_model, dataloader)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
