"""
testing command line:
python ./Scripts/CrossModalityGeneration/UniTCR_testing_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml --input ./Data/Examples/Example_CMG_test_TCRs.csv
"""
import scanpy as sc
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import sys
sys.path.append(f"{os.getcwd()}/Scripts/Pretrain/")
import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import joblib
import pandas as pd
from InputDataSet_CrossModalityGeneration import InputDataSet
from torch.nn import functional as F
from UniTCR_Modelblocks_CrossModalityGeneration import Decoder_model
import logging
import yaml
import anndata as ad
from TCR_encoding import TCR_encoding
from ModelBlocks_UniTCR_pretrain import UniTCR_model
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-C","--config", required = True, help = "path the config file")
ap.add_argument("-i","--input", required = True, help = "path to the testing file")
args = ap.parse_args()

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

# load a trained decoder model
Decoder_model_example = Decoder_model(compressed_TCR_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                                      compressed_profile_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'],
                                      decoderprofile_in_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'], 
                                      decoderprofile_hid_dim = config['Train']['Model_Parameter']['encoderprofile_hid_dim'], 
                                      decoderprofile_hid_dim2 = config['Train']['Model_Parameter']['encoderprofile_hid_dim2'],
                                      decoderprofile_out_dim = config['Train']['Model_Parameter']['encoderprofle_out_dim'],
                                      ).to(device)

checkpoint = torch.load(config['Inference']['Model'])
Decoder_model_example.load_state_dict(checkpoint['model_state_dict'])


# load a pretrained UniTCR model
UniTCR_example = UniTCR_model(encoderTCR_in_dim = config['Train']['Model_Parameter']['encoderTCR_in_dim'],
                              encoderTCR_out_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                              encoderprofile_in_dim = config['Train']['Model_Parameter']['encoderprofle_out_dim'], 
                              encoderprofile_hid_dim = config['Train']['Model_Parameter']['encoderprofile_hid_dim2'], 
                              encoderprofile_hid_dim2 = config['Train']['Model_Parameter']['encoderprofile_hid_dim'],
                              encoderprofle_out_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'],
                              out_dim = config['Train']['Model_Parameter']['out_dim'],
                              head_num = config['Train']['Model_Parameter']['head_num'],
                              ).to(device)

checkpoint = torch.load("./Requirements/Pretrained_model.pth")
UniTCR_example.load_state_dict(checkpoint['model_state_dict'])

testing_data = pd.read_csv(args.input)
TCRs = np.array(testing_data['TCR'])
TCR_encodings = [TCR_encoding(i) for i in TCRs]
TCR_encodings = torch.stack(TCR_encodings, axis = 0)


# prediction started
with torch.no_grad():
    Profile_pre = []
    for i in range(int(TCR_encodings.shape[0] / 10000) + 1):
        tmp = TCR_encodings[i * 10000:(i+1)*10000]
        encoderTCR_feature, mask_TCR = UniTCR_example.encoder_TCR.eval()(tmp.to(device))
        encoderTCR_embedding = encoderTCR_feature.sum(dim = 1) * (1 / (~mask_TCR).sum(dim = 1)).unsqueeze(0).T
        encoderTCR_embedding = UniTCR_example.TCR_proj.eval()(encoderTCR_embedding)
        encoderTCR_embedding = F.normalize(encoderTCR_embedding, dim = -1)
        profile_embedding_pre = Decoder_model_example.TCR2profile.eval()(encoderTCR_embedding)
        profile_pre = Decoder_model_example.decoder_profile.eval()(profile_embedding_pre)
        Profile_pre.append(profile_pre.cpu())
Profile_pre = torch.cat(Profile_pre, axis = 0)

generation_profile = ad.AnnData(Profile_pre.numpy())
generation_profile.obs['beta'] = TCRs
generation_profile.var['gene'] = np.array(pd.read_csv("./Requirements/CMG_genes_5000.csv")['genes'])
generation_profile.write(f"{config['Train']['output_dir']}/Generation_result.h5ad")
