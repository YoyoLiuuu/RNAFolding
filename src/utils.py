import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence


import fm
import random
from Bio import SeqIO

_, alphabet = fm.pretrained.rna_fm_t12()

class RNADataset(torch.utils.data.Dataset): 
    def __init__(self, alphabet, fasta_path = os.path.join("Ribozyme_data", "ribocentre.fasta"), masking_ratio = 0.3):
        self.rna_list = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            self.rna_list.append(record.seq)
        self.alphabet = alphabet        
        self.masking_ratio = masking_ratio
            
    def __len__(self, ):
        return len(self.rna_list)
    
    def __getitem__(self, idx):
        rna_seq = self.rna_list[idx]
        rna_token = torch.empty(len(rna_seq)) 
        
        rna_token = torch.tensor([self.alphabet.get_idx(s) for s in rna_seq])
        
        masked, unmasked = RNADataset.RNAMasker(rna_token, self.masking_ratio, k_mer = 3)
        
        
        if self.alphabet.prepend_bos:
            masked = torch.cat((torch.tensor([self.alphabet.cls_idx]), masked))
            unmasked = torch.cat((torch.tensor([self.alphabet.cls_idx]), unmasked))
        
        if self.alphabet.append_eos:
            masked = torch.cat((masked, torch.tensor([self.alphabet.eos_idx])))
            unmasked = torch.cat((unmasked, torch.tensor([self.alphabet.eos_idx])))
        
        return (masked, unmasked)
    
    @staticmethod
    def RNAMasker(seq, masking_ratio, k_mer): 
        """
        We mask array continguously. This allows the model to learn
        RNA representations effectively. We first pick a % of the array
        to mask. 
        
        Then identify a random int and then mask from there for x token ids.
        
        We leave a k-mer, however the sequence is tokenized as a 1-mer,
        at the start and end of the RNA sequence. This is to ensure
        that the model learns an effective representation.
        """
        
        masked_seq = seq.detach().clone()
        masking_count = masked_seq.shape[0]*masking_ratio
        start_idx = random.randint(k_mer, masked_seq.shape[0] - (k_mer+int(masking_count)))

        end_idx = start_idx + int(masking_count)
        
        for i in range(start_idx, end_idx):
            masked_seq[i] = 24
        return (masked_seq, seq)
    

def custom_collate(batch):
    """
    collate_fn to deal with inputs being
    different sizes. This basically pads with
    <pad> tokens.
    """
    
    masked = [seq[0] for seq in batch]
    unmasked = [seq[1] for seq in batch]

    
    masked_pad = pad_sequence(masked, batch_first = True, padding_value=1)
    unmasked_pad = pad_sequence(unmasked, batch_first = True, padding_value=1)
    
    lengths = torch.tensor([len(seq) for seq in masked])
    
    return masked_pad, unmasked_pad, lengths

dataset = RNADataset(alphabet=alphabet)
dataloader = DataLoader(dataset, batch_size=5, collate_fn=custom_collate)

print(dataloader)