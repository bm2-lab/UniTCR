# UniTCR
## Introduction 
UniTCR is a unified framework composed of a dual-modality contrastive learning module and a modality preservation module. This model can be easily adapted to four applications in the computational immunology, including:
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
#### * Note : you should install CUDA and cuDNN version compatible with the pytorch version [Version Searching](https://pytorch.org/). 
## Usage  
### 1. Single modality analysis / Modality gap analysis
#### Pretrain

    python ./Scripts/Pretrain/UniTCR_pretrain.py --config ./Configs/TrainingConfig_pretrain.yaml
    
#### Single modality embedding extraction / modality gap calculation

    python ./Scripts/Pretrain/Embedding_extraction.py --config ./Configs/TrainingConfig_pretrain.yaml

### 2. Epitope-TCR binding prediction
#### 2.1 No HLA information
#### Training:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml
#### Testing:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing.csv

#### 2.2 Incorporating HLA information
#### Training:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre_HLA.py  --config ./Configs/TrainingConfig_EpitopeBindPrediction_HLA.yaml
#### Testing:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre_HLA.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing_HLA.csv
### 3. Cross-modaltiy generation
#### Training:

    python ./Scripts/CrossModalityGeneration/UniTCR_training_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml
#### Testing:

    python ./Scripts/CrossModalityGeneration/UniTCR_testing_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml --input ./Data/Examples/Example_CMG_test_data.h5ad

## Citation

## Contacts
bm2-lab@tongji.edu.cn
