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
### Pretrain

    python ./Scripts/Pretrain/UniTCR_pretrain.py --config ./Configs/TrainingConfig_pretrain.yaml
#### Single modality embedding extraction and modality gap calculation

    python ./Scripts/Pretrain/Embedding_extraction.py --config ./Configs/TrainingConfig_pretrain.yaml

### Epitope-TCR binding prediction
#### No HLA information
training:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml
testing:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing.csv

#### incorporating HLA information
training:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre_HLA.py  --config ./Configs/TrainingConfig_EpitopeBindPrediction_HLA.yaml
testing:

    python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre_HLA.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing_HLA.csv
### Cross-modaltiy generation
## Citation

## Contacts
bm2-lab@tongji.edu.cn
