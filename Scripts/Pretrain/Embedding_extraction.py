"""
embedding extraction command line:
python ./Scripts/Pretrain/Embedding_extraction.py --config ./Configs/TrainingConfig_pretrain.yaml
"""

import scanpy as sc
import anndata as ad
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import joblib
import pandas as pd
from InputDataSet_pretrain import InputDataSet
from torch.nn import functional as F
from ModelBlocks_UniTCR_pretrain import UniTCR_model
import logging
import yaml
import argparse 

argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, help='the path to the config file (*.yaml)',required=True)
args = argparser.parse_args()

# load the cofig file
config_file = args.config
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file)

# random seed setting
torch.manual_seed(config['Train']['Trainer_parameter']['random_seed'])
torch.cuda.manual_seed_all(config['Train']['Trainer_parameter']['random_seed'])
np.random.seed(config['Train']['Trainer_parameter']['random_seed'])
random.seed(config['Train']['Trainer_parameter']['random_seed'])
torch.cuda.manual_seed(config['Train']['Trainer_parameter']['random_seed'])
device = config['Train']['Model_Parameter']['device']

if not os.path.exists(config['Train']['output_dir']+'/Embedding_Result'):
    os.makedirs(config['Train']['output_dir']+'/Embedding_Result')

# initialize a new model
UniTCR_example = UniTCR_model(encoderTCR_in_dim = config['Train']['Model_Parameter']['encoderTCR_in_dim'],
                              encoderTCR_out_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                              encoderprofile_in_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'], 
                              encoderprofile_hid_dim = config['Train']['Model_Parameter']['encoderprofile_hid_dim'], 
                              encoderprofile_hid_dim2 = config['Train']['Model_Parameter']['encoderprofile_hid_dim2'],
                              encoderprofle_out_dim = config['Train']['Model_Parameter']['encoderprofle_out_dim'],
                              out_dim = config['Train']['Model_Parameter']['out_dim'],
                              head_num = config['Train']['Model_Parameter']['head_num'],
                              ).to(device)

checkpoint = torch.load(config['Inference']['Model'])
UniTCR_example.load_state_dict(checkpoint['model_state_dict'])


testing_data = InputDataSet(config['dataset']['Training_dataset'])

# loading dataloader
dataloader1 = torch.utils.data.DataLoader(testing_data, 10000, shuffle = False)


# prediction started
with torch.no_grad():
    TCR_embedding = []
    Profile_embedding = []
    Gaps = []
    for idx, b in enumerate(dataloader1):
        encoderTCR_feature, mask_TCR = UniTCR_example.encoder_TCR.eval()(b[0].to(device))
        encoderprofile_feature = UniTCR_example.encoder_profile.eval()(b[1].to(torch.float32).to(device))

        encoderTCR_embedding = encoderTCR_feature.sum(dim = 1) * (1 / (~mask_TCR).sum(dim = 1)).unsqueeze(0).T
        encoderTCR_embedding = UniTCR_example.TCR_proj.eval()(encoderTCR_embedding)
        encoderTCR_embedding = F.normalize(encoderTCR_embedding, dim = -1)
        
        encoderprofile_embedding = UniTCR_example.profile_proj.eval()(encoderprofile_feature)
        encoderprofile_embedding = F.normalize(encoderprofile_embedding, dim = -1)
        
        TCR_embedding.append(encoderTCR_embedding.cpu())
        Profile_embedding.append(encoderprofile_embedding.cpu())
        tmp_gap = 2 * (1 - (encoderTCR_embedding @ encoderprofile_embedding.T).diag())

        Gaps.append(tmp_gap.cpu())
        
    TCR_embedding = torch.cat(TCR_embedding, dim = 0)
    Profile_embedding = torch.cat(Profile_embedding, dim = 0)
    Gaps = torch.cat(Gaps, dim = 0)
    
original_data = sc.read_h5ad(config['dataset']['Training_dataset'])

adata_TCR = ad.AnnData(TCR_embedding.numpy())
adata_TCR.obs = original_data.obs
adata_TCR.write(f"{config['Train']['output_dir']}/Embedding_Result/TCR_embedding.h5ad")

adata_profile = ad.AnnData(Profile_embedding.numpy())
adata_profile.obs = original_data.obs
adata_profile.write(f"{config['Train']['output_dir']}/Embedding_Result/Profile_embedding.h5ad")

adata_gaps = ad.AnnData(Gaps.unsqueeze(1).numpy())
adata_gaps.obs = original_data.obs
adata_gaps.write(f"{config['Train']['output_dir']}/Embedding_Result/Gaps.h5ad")
