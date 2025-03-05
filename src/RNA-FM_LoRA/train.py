import os
import torch
import torch.nn as nn
from tqdm import tqdm # Not really needed but I liked to see bar go brr

from utils import *

def train(model, dataloader, checkpoint_dir,  num_epochs=10, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=1) # 1 because of the padded tokens
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            
            masked_sequences, unmasked_sequences, _ = batch
            masked_sequences = masked_sequences.to(device).long()
            unmasked_sequences = unmasked_sequences.to(device).long()
            
            outputs = model(masked_sequences)["logits"] # The right format.
            loss = criterion(outputs.view(-1, outputs.size(-1)), unmasked_sequences.view(-1))
            
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
        print(f"{epoch + 1}: {epoch_loss / len(dataloader)}")

        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }, checkpoint_path)
        print(f"Saved best model checkpoint to {checkpoint_path}")

    return None
