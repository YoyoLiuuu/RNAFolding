# RNA Folding

## Background
RNA molecules are crucial compoments in the central dogma of biology. Not only are they the intermediary messengers of all genetic information in a cell, but they have a whole host of functions ranging from regulating gene expression, to being the primary genetic material for viruses. Yet, our understanding of these vital molecules are limited by our current methods, namely the inability to model RNA. RNA when synthesized in the cell immediately folds up into a tertiary/three dimensional structure, creating a "landscape" of potential folding pathways and configurations. Studying these structures are crucial for understanding one of the most important molecules in an organism, enabling faster drug discovery processes, better vaccines, and more. Solving this problem would instantly revolutionize the world, making this an extremely pressing problem. To address this very problem, the RNA Folding team is developing a state-of-the-art model for RNA sequence to tertiary structure prediction. 

## Methods

Talk about ribozymes and why we are choosing this

talk about the data from logan (SRA wide ribozyme search)

transformer model

GNN model

pdb structures and potentially augementing it


## About the Team
Team co-leads: Purav Gupta, Yoyo Liu
Members: Nicholas Carbones, Xin Lei Lin, Flora Liu, Ahmad Khan, Rahul Selvakumar

To run the code you need to setup a venv (if you want lol) and then run the requirements.txt file. 

Then call:
python src/main.py --data_path ./rna_sequences --checkpoint_dir ./lora_checkpoints --epochs 1 --batch_size 2

from the dir of the repo. It probably won't work all the way because of LoRA and how it is setup. LoRA commented out is the og LoRA and current implementation is my custom code. 