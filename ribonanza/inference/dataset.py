import pandas as pd 
import torch 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import random 

from torch.utils.data import Dataset, DataLoader 

# modified from code by Shujun 

class RNA2D_set(Dataset): 
    def __init__(self, data): 
        self.data = pd.DataFrame(data) # convert input to DF
        self.tokens = {nt:i for i, nt in enumerate('ACGU')}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = [self.tokens[nt] for nt in self.data.loc[idx, 'sequence']]
        sequence = np.array(sequence)
        sequence = torch.tensor(sequence)

        return {'sequence': sequence}
