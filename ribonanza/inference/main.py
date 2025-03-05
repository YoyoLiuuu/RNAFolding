from dataset import RNA2D_set
from model import finetuned_RibonanzaNet
from model import load_config_from_yaml
import pandas as pd 

test_data = pd.read_csv('data/casp15.csv')
test_dataset = RNA2D_set(test_data)

print(test_dataset[0])

model = finetuned_RibonanzaNet(load_config_from_yaml("ribonanzanet2d/configs/pairwise.yaml")).cuda()

# model.load_state_dict(torch.load(''))