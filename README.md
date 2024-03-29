# UniTCR
[![DOI image](https://zenodo.org/badge/DOI/10.5281/zenodo.10891094.svg)](https://zenodo.org/record/10891094#.Y8a7dXZBxHU)
## Introduction 
UniTCR is a novel low-resource-aware multi-modal representation learning framework for unifiably integration and joint analysis of T cell receptor (TCR) and its corresponding transcriptome, which is composed of a dual-modality contrastive learning module and a modality preservation module. This model can be easily adapted to four applications in the computational immunology, including:
* Single modality analysis
* Modality gap analysis
* Epitope-TCR binding prediction
* Cross-modality generation  
<p align="center">
<img src="https://github.com/bm2-lab/UniTCR/assets/89248357/943b3549-59a5-4367-b599-d5abaf0df3b0" width = 80%></img>
</p>

## Table of contents
- [Requirements](https://github.com/bm2-lab/UniTCR/tree/main#requirements)
- [Installation](https://github.com/bm2-lab/UniTCR/tree/main#installation)
- [Usage](https://github.com/bm2-lab/UniTCR/tree/main#Usage)
  - [Single modality analysis / Modality gap analysis](https://github.com/bm2-lab/UniTCR/blob/main/README.md#1-single-modality-analysis--modality-gap-analysis)
    - [Pretrain](https://github.com/bm2-lab/UniTCR/tree/main#pretrain)
    - [Single modality embedding extraction / modality gap calculation](https://github.com/bm2-lab/UniTCR/tree/main#single-modality-embedding-extraction--modality-gap-calculation)
  - [Epitope-TCR binding prediction](https://github.com/bm2-lab/UniTCR/tree/main#2-epitope-tcr-binding-prediction)
    - [No HLA information](https://github.com/bm2-lab/UniTCR/tree/main#21-no-hla-information)
      - [Training](https://github.com/bm2-lab/UniTCR/tree/main#training)
      - [Testing](https://github.com/bm2-lab/UniTCR/tree/main#testing)
    - [Incorporating HLA information](https://github.com/bm2-lab/UniTCR/tree/main#22-incorporating-hla-information)
      - [Training](https://github.com/bm2-lab/UniTCR/tree/main#training-1)
      - [Testing](https://github.com/bm2-lab/UniTCR/tree/main#testing-1)
  - [Cross-modaltiy generation](https://github.com/bm2-lab/UniTCR/tree/main#3-cross-modaltiy-generation)
    - [Training](https://github.com/bm2-lab/UniTCR/tree/main#training-2)
    - [Testing](https://github.com/bm2-lab/UniTCR/tree/main#testing-2)
- [Citation](https://github.com/bm2-lab/UniTCR/tree/main#citation)
- [Contacts](https://github.com/bm2-lab/UniTCR/tree/main#contacts)
  
## Requirements  
* python == 3.9.7  
* pytorch == 1.10.2  
* numpy == 1.21.2  
* pandas == 1.4.1  
* scipy == 1.7.3
* scanpy == 1.9.1
* anndata == 0.8.0
#### * Note : you should install CUDA and cuDNN version compatible with the pytorch version [Version Searching](https://pytorch.org/). 

## Installation
#### 1. Downloading UniTCR

    git clone https://github.com/bm2-lab/UniTCR.git

#### 2. Downloading example data and example output of UNiTCR
Due to the storage limit of git repository, we did not upload our example data and output files of these example data. Instead, we have uploaded the [example data](https://drive.google.com/drive/folders/1EcU9dDqjNBGOBt1QrccWMswLrqXzGjcF?usp=drive_link) and their corresponding [output files](https://drive.google.com/drive/folders/1WVJIf-u7jocQp7L0DeIlizvz5wbndS2w?usp=drive_link) to the google drive. The example data contains three directories, including "Examples", "EpitopeTCRBinding", and "EpitopeTCRBinding_HLA". The "Examples" directory constains all the example data to make sure running UniTCR correctly. The "EpitopeTCRBinding" directory contains the 5-fold train/validation/testing data used to evaluate the performance of models in three testing scenarios, i.e. majority testing, few-shot testing and zero-shot testing. The "EpitopeTCRBinding_HLA" directory contains the 5-fold train/validation/testing data used to evaluate the performance of models with incorporating HLA information in three testing scenarios. So please download these files to make sure you can use UniTCR to perform analysis correctly. After downloading, please add [example data](https://drive.google.com/drive/folders/1EcU9dDqjNBGOBt1QrccWMswLrqXzGjcF?usp=drive_link) to the "Data" directory and [output files](https://drive.google.com/drive/folders/1WVJIf-u7jocQp7L0DeIlizvz5wbndS2w?usp=drive_link) to the "Experiments" directory in our git repository.  
Please note that our test was carried out on a machine equipped with four NVIDIA GeForce RTX 3090 GPUs. Hence, it's essential to configure the default setting of `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")` according to your machine's specifications before executing our scripts.
## Usage  
### 1. Single modality analysis / Modality gap analysis
#### Pretrain
Command:

    python ./Scripts/Pretrain/UniTCR_pretrain.py --config ./Configs/TrainingConfig_pretrain.yaml
* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_Pretrain". These files are the records for the model training.
#### Single modality embedding extraction / modality gap calculation
Command：

    python ./Scripts/Pretrain/Embedding_extraction.py --config ./Configs/TrainingConfig_pretrain.yaml

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Embedding_Result" in the directory "./Experiments/TrainingResult_Pretrain", which contains a profile embedding (* .h5ad), a TCR embedding (* .h5ad) and a gap information for each T cell (* .h5ad)
### 2. Epitope-TCR binding prediction
#### 2.1 No HLA information 
This is the setting that the TCR encoder of UniTCR is used for constructing the epitope-TCR binding prediction model without using HLA information.
#### Training:
Command

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml

* config.yaml: input *.yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_BindPre". These files are the records for the model training.
#### Testing:
Command：

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing.csv

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".
* input.csv: input * .csv file contains three columns: Beta, Peptide and Label, which represents TCR CDR3 sequence, the epitope sequence,  and their binding specificity.
In the Label column, there are two values: 1 indicating binding, 0 indicating non-binding.

This command line will output a "Prediction_result.csv" in the directory "./Experiments/TrainingResult_BindPre". This file contains four columns:  Beta, Peptide, Label, and Rank, which represents TCR CDR3 sequence, the epitope sequence, the ground-truth binding specificity, and their predicted binding score, respectively. 

#### 2.2 Incorporating HLA information
This is the setting that the TCR encoder of UniTCR is used for constructing the epitope-TCR binding prediction model using HLA information.
#### Training:
Command：

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre_HLA.py  --config ./Configs/TrainingConfig_EpitopeBindPrediction_HLA.yaml

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_BindPre_HLA". These files are the records for the model training.
#### Testing:
Command：

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre_HLA.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing_HLA.csv 

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".
* input.csv: input * .csv file contains three columns: Beta, Peptide, HLA and Label, which represents TCR CDR3 sequence, the epitope sequence, HLA information and their binding specificity.
In the Label column, there are two values: 1 indicating binding, 0 indicating non-binding.

This command line will output a "Prediction_result.csv" in the directory "./Experiments/TrainingResult_BindPre_HLA". This file contains four columns:  Beta, HLA, Peptide, Label, and Rank, which represents TCR CDR3 sequence, HLA information, the epitope sequence, the ground-truth binding specificity, and their predicted binding score, respectively. 
### 3. Cross-modaltiy generation
#### Training:
Command：

    python ./Scripts/CrossModalityGeneration/UniTCR_training_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_CrossModalityGeneration". These files are the records for the model training.
#### Testing:
Command：

    python ./Scripts/CrossModalityGeneration/UniTCR_testing_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml --input ./Data/Examples/Example_CMG_test_TCRs.csv

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, model trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".
* input.csv: input * .csv file contains one column: TCR, which represents TCR CDR3 sequence.

This command line will output a "Generation_result.h5ad" in the directory "./Experiments/TrainingResult_CrossModalityGeneration". This file contains the predicted T cell transcriptome.
## Citation
Yicheng Gao, Kejing Dong, Qi Liu et al. *Unified cross-modality integration and analysis of T-cell receptors and T-cell transcriptomes*, biorxiv, 2023.
## Contacts
bm2-lab@tongji.edu.cn
