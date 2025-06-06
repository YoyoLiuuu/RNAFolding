Ribonanza: 
For our project, on EC2 instance, commands to install conda, set up environment, and run the code is: 
Set up conda: 
1. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
2. bash ~/miniconda.sh -b -p $HOME/miniconda
3. ~/miniconda/bin/conda init bash
4. source ~/.bashrc

Create environment: 
1. conda env create -f env.yml

Activate: 
1. conda activate torch 

GPU set up: 
1. lspci | grep -i nvidia
2. sudo apt update
3. sudo apt install -y nvidia-driver-535
4. CAREFULE: sudo reboot 

tmscoring install 
1. go into the folder (where setup.py is located)
2. pip install . 

# RNA Folding

## Background
RNA molecules are crucial components in the central dogma of biology. Not only are they the intermediary messengers of all genetic information in a cell, but they have a host of functions ranging from regulating gene expression to being the primary genetic material for viruses. Yet, our understanding of these vital molecules is limited by our current methods—experimentally, crystallizing an RNA structure is challenging; computationally, modelling its structure is computate intensive. 

RNA when synthesized in the cell immediately folds up into a tertiary/three-dimensional structure, creating a "landscape" of potential folding pathways and configurations. These structures have important functionalities in the body from catalyzing reactions to decoding genetic information. Understanding their structure can provide insights to fields like precision medicine and vaccine development. At UTMIST, the RNA Folding team is developing a state-of-the-art model for RNA sequence to tertiary structure prediction.
 
## Methods
There are several overarching classes of RNA molecules that represent a plethora of RNA sequences. To start, we chose the class of RNA molecules called Ribozymes, a special type of RNA molecule that has the ability to catalyze specific biochemical reactions. Ribozymes were chosen due to their diversity in structure and discovery potential. According to a widely accepted theory, the RNA World Hypothesis, ribozymes are a modern descendant of the very first cellular replicator. Thus, by  studying Ribozymes, we hope to develop an accurate model that has the potential to be generalized and applied to other molecules.  

Currently, the biggest bottleneck in RNA Folding models is the sparsity of data. There are only ~1000 high-quality RNA structures in public databases, making it very easy to overfit to these limited data points. However, by focusing on Ribozymes and expanding our training datasets to other types of data, such as sequence data, we can mine the biggest public repository of sequence data in the world: the SRA. By mining this dataset, we aim to supplement our ~200 structures (after sub-setting for only ribozyme structures) with hundreds of thousands of ribozyme sequences. This primary sequence data represents a linear chain structure of an RNA molecule. This representation doesn’t hold much information, but by creating an MSA of hundreds of sequences, the covariance between positions can be calculated creating an information dense matrix of sequences, apt for predicting RNA structure. This type of large-scale data mining approach combined with an end to end deep learning platform has never been done before for ribozymes, representing a huge opportunity for novelty and improvement.
 
Every 2 years, a biennial competition called the CASP competition runs, where teams of scientists from all across the world submit their best structure prediction models for testing on new Protein and RNA structures. Ever since AlphaFold entered the competition in 2018 and AlphaFold 2 in 2020, the world of structure prediction modelling has exploded. The popularity of this field was only furthered by the bestowment of a Nobel Prize to the inventors of AlphaFold. For the past 2 iterations of CASP, they have expanded the competition to include several different categories ranging from single Proteins, short RNA molecules, RNA-RNA complexes, RNA-Protein complexes and more. The RNA molecules category has garnered hundreds of submissions with models of all types ranging from end-to-end deep learning models to physics and chemistry-based models. After several months of research and discussion, we as a team settled on a model called RhoFold+, published in November 2024, which boasted the best performance on RNA testing metrics, when compared to its rival structure prediction methods. This end-to-end deep learning model features a language model as well as a transformer block for accurate RNA structure prediction. Thus, we as a team decided to use this model as our baseline to improve on.
 
Making the SOTA of any field requires a deep understanding of the field and its requirements. Thus, our first aim was to thoroughly review the RNA Folding literature, upon which we chose RhoFold+ as our base model for comparison and improvement. We first plan to fine-tune RNA-FM or the language model which they created for RhoFold+. This language model is originally based on ESM-RNA (a famous and accurate RNA language model) and fine-tuned for RNA structure prediction. We are further fine-tuning this language model for ribozyme-specific prediction, using the thousands of sequences that we mined from the SRA (the largest public RNA sequence dataset), along with a new proprietary dataset that was only publicly released a few months ago called Ribonanza. With these two, new and information-dense sources, the language model should be able to create accurate embeddings given any RNA Ribozyme sequence. Furthermore, we are planning on fine-tuning the rest of the transformer blocks, by possibly adding on a few more layers while keeping the rest of the transformer layers frozen.
 
Finally, we will also be augmenting the ~200 accurate RNA structures that we have. These structures were experimentally generated and graciously deposited into the PDB (Protein Data Bank) by researchers across the world. Each structure was painstakingly made and takes several months at least to build the crystals and solve the structures. However, with the limited nature of these sequences, we need to augment these structures to create a larger diversity of training data for our model. There are some ways to augment them, from augmenting the RNA sequence that represents that structure, to possibly adding noise to the 3D structure representation itself. We are currently looking into more ways to augment these limited structures and hopefully generate better predictions.
 
Overall, this project is operating at the forefront of RNA Ribozyme biology and by combining several different features, such as newly mined Ribozyme sequences, the Ribonanza data, and by fine-tuning the SOTA RNA structure predictor, we aim to improve on the SOTA and create the best Ribozyme tertiary structure prediction model. After accomplishing this, our next aim will be to expand out to other classes of RNA molecules, such as Riboswitches, tRNA, and more.
 
## About the Team
Team co-leads: Purav Gupta, Yoyo Liu
Members: Nicholas Carbones, Xin Lei Lin, Flora Liu, Ahmad Khan, Rahul Selvakumar

To run the code you need to setup a venv (if you want lol) and then run the requirements.txt file. 

Then call:
python src/main.py --data_path ./rna_sequences --checkpoint_dir ./lora_checkpoints --epochs 1 --batch_size 2

from the dir of the repo. It probably won't work all the way because of LoRA and how it is setup. LoRA commented out is the og LoRA and current implementation is my custom code. 