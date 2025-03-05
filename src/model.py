import fm
from src.lora import add_lora
import pytorch_lightning as pl
import torch 
import torch.nn as nn
from tqdm import tqdm 

def load_rna_fm_t12():
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = add_lora(model)
    return model, alphabet

class RNAFMLitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)["logits"]
    
    def training_step(self, batch, batch_idx):
        print("HI")
        masked_sequences, unmasked_sequences, _ = batch
        masked_sequences = masked_sequences.long()
        unmasked_sequences = unmasked_sequences.long()
        
        outputs = self(masked_sequences)
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), unmasked_sequences.view(-1))
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss  # Removed tqdm

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}