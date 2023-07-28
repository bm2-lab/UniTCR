# UniTCR
## Introduction 
UniTCR is a unified framework for integration and joint analysis of T cell receptor (TCR) and its corresponding transcriptome, which is composed of a dual-modality contrastive learning module and a modality preservation module. This model can be easily adapted to four applications in the computational immunology, including:
* Single modality analysis
* Modality gap analysis
* Epitope-TCR binding prediction
* Cross-modality generation

## Requirements  
* python == 3.9.7  
* pytorch == 1.10.2  
* numpy == 1.21.2  
* pandas == 1.4.1  
* scipy == 1.7.3
* scanpy == 1.9.1
* anndata == 0.8.0
#### * Note : you should install CUDA and cuDNN version compatible with the pytorch version [Version Searching](https://pytorch.org/). 
## Usage  
### 1. Single modality analysis / Modality gap analysis
#### Pretrain
Command:

    python ./Scripts/Pretrain/UniTCR_pretrain.py --config ./Configs/TrainingConfig_pretrain.yaml
* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_Pretrain". These files are the records for the model training.
#### Single modality embedding extraction / modality gap calculation
Command：

    python ./Scripts/Pretrain/Embedding_extraction.py --config ./Configs/TrainingConfig_pretrain.yaml

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Embedding_Result" in the directory "./Experiments/TrainingResult_Pretrain", which contains a profile embedding (* .h5ad), a TCR embedding (* .h5ad) and a gap information for each T cell (* .h5ad)
### 2. Epitope-TCR binding prediction
#### 2.1 No HLA information 
This is the setting that the TCR encoder of UniTCR is used for constructing the epitope-TCR binding prediction model without using HLA information.
#### Training:
Command

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml

* config.yaml: input *.yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_BindPre". These files are the records for the model training.
#### Testing:
Command：

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing.csv

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".
* input.csv: input * .csv file contains three columns: Beta, Peptide and Label, which represents TCR CDR3 sequence, the epitope sequence,  and their binding specificity.
In the Label column, there are two values: 1 indicating binding, 0 indicating non-binding.

This command line will output a "Prediction_result.csv" in the directory "./Experiments/TrainingResult_BindPre". This file contains four columns:  Beta, Peptide, Label, and Rank, which represents TCR CDR3 sequence, the epitope sequence, the ground-truth binding specificity, and their predicted binding score, respectively. 

#### 2.2 Incorporating HLA information
This is the setting that the TCR encoder of UniTCR is used for constructing the epitope-TCR binding prediction model using HLA information.
#### Training:
Command：

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre_HLA.py  --config ./Configs/TrainingConfig_EpitopeBindPrediction_HLA.yaml

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_BindPre_HLA". These files are the records for the model training.
#### Testing:
Command：

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre_HLA.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing_HLA.csv 

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".
* input.csv: input * .csv file contains three columns: Beta, Peptide, HLA and Label, which represents TCR CDR3 sequence, the epitope sequence, HLA information and their binding specificity.
In the Label column, there are two values: 1 indicating binding, 0 indicating non-binding.

This command line will output a "Prediction_result.csv" in the directory "./Experiments/TrainingResult_BindPre_HLA". This file contains four columns:  Beta, HLA, Peptide, Label, and Rank, which represents TCR CDR3 sequence, HLA information, the epitope sequence, the ground-truth binding specificity, and their predicted binding score, respectively. 
### 3. Cross-modaltiy generation
#### Training:
Command：

    python ./Scripts/CrossModalityGeneration/UniTCR_training_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".

This command line will output a directory "Model_checkpoints" and a "training.log" in the directory "./Experiments/TrainingResult_CrossModalityGeneration". These files are the records for the model training.
#### Testing:
Command：

    python ./Scripts/CrossModalityGeneration/UniTCR_testing_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml --input ./Data/Examples/Example_CMG_test_TCRs.csv

* config.yaml: input * .yaml file contains all necessary parameters that used for UniTCR, containing model parameters, modol trained for inference, and the output directory, etc. Detailed information can be found in the directory "./Configs".
* input.csv: input * .csv file contains one column: TCR, which represents TCR CDR3 sequence.

This command line will output a "Generation_result.h5ad" in the directory "./Experiments/TrainingResult_CrossModalityGeneration". This file contains the predicted T cell transcriptome.
## Citation

## Contacts
bm2-lab@tongji.edu.cn
