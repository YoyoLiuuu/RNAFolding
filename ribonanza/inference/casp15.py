# this is a script using RibonanzaNet to predict secondary structure for CASP 15

from dataset import RNA2D_set
from model import finetuned_RibonanzaNet
from model import load_config_from_yaml
import pandas as pd 
from tqdm import tqdm 
import torch 
from hungarian import mask_diagonal 
from arnie.pk_predictors import _hungarian 
import numpy as np


# load test data 
test_data = pd.read_csv('data/casp15.csv')
test_dataset = RNA2D_set(test_data)

print(test_dataset[0])

model = finetuned_RibonanzaNet(load_config_from_yaml("ribonanzanet2d/configs/pairwise.yaml")).cuda()

model.load_state_dict(torch.load('weight/RibonanzaNet-SS.pt', map_location='cpu'))

# making predictions 

test_preds = []
model.eval()

# track execution/progress bar 
for i in tqdm(range(len(test_dataset))):
    example = test_dataset[i]
    sequence = example['sequence'].cuda().unsqueeze(0)

    with torch.no_grad():
        test_preds.append(model(sequence).sigmoid().cpu().numpy())

test_preds_hungarian=[]
hungarian_structures=[]
hungarian_bps=[]

for i in range(len(test_preds)):

    s,bp=_hungarian(mask_diagonal(test_preds[i][0]),theta=0.5,min_len_helix=1) #best theta based on val is 0.5
    hungarian_bps.append(bp)
    ct_matrix=np.zeros((len(s),len(s)))

    for b in bp:
        ct_matrix[b[0],b[1]]=1

    ct_matrix=ct_matrix+ct_matrix.T
    test_preds_hungarian.append(ct_matrix)
    hungarian_structures.append(s)

print(hungarian_structures[0])