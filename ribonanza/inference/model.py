import sys 
import torch 

sys.path.append("ribonanzanet2d")

from ribonanzanet2d.Network import * 
import yaml 

class Config: 
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)
    
def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file: 
        config = yaml.safe_load(file)
    return Config(**config)
    
class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        config.dropout = 0.3 
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.dropout = nn.Dropout(0.0)
        self.ct_predictor = nn.Linear(64, 1)
    
    def forward(self, src):
        # torch.no_grad():
        _, pairwise_features = self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        pairwise_features = pairwise_features + pairwise_features.permute(0, 2, 1, 3) # symetry

        output = self.ct_predictor(self.dropout(pairwise_features))

        return output.squeeze(-1)
