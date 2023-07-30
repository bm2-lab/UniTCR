"""
testing command line:
python ./Scripts/EpitopeBindingPrediction/UniTCR_Testing_BindPre.py --config ./Configs/TrainingConfig_EpitopeBindPrediction.yaml --input ./Data/Examples/Example_testing.csv
"""
import scanpy as sc
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import joblib
import pandas as pd
from torch.nn import functional as F
from Modelblocks_UniTCR_BindPre import UniTCR_model
import logging
import yaml
from pMHC_TCR_encoding import pMHC_TCR_encoding
from InputDataSet_Testing import InputDataSet_val
import time
from pMHC_TCR_encoding import TCR_encoding
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-C","--config", required = True, help = "path to the config file")
ap.add_argument("-i","--input", required = True, help = "path to the testing file")
args = ap.parse_args()

# load the cofig file
config_file = args.config
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file)

device = config['Train']['Model_Parameter']['device']

# load a trained UniTCR model
UniTCR_example = UniTCR_model(encoderTCR_in_dim = config['Train']['Model_Parameter']['encoderTCR_in_dim'],
                              encoderTCR_out_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                              encoderpMHC_in_dim = config['Train']['Model_Parameter']['encoderpMHC_in_dim'],
                              encoderpMHC_out_dim = config['Train']['Model_Parameter']['encoderpMHC_out_dim'],
                              out_dim = config['Train']['Model_Parameter']['out_dim'],
                              head_num = config['Train']['Model_Parameter']['head_num'],
                              ).to(device)
checkpoint = torch.load(config['Inference']['Model'])
UniTCR_example.load_state_dict(checkpoint['model_state_dict'])

# HLA type
# loading testing dataset
testing_data = pd.read_csv(args.input)

################ make prediction ################# 

# construct the list for storing the rank results
rank_output = []

# construct the dataset
dataset1 = InputDataSet_val(testing_data)

# loading dataloader with 10001 batch
dataloader1 = torch.utils.data.DataLoader(dataset1, 10000, shuffle = False)

# prediction started
with torch.no_grad():
    rank_output = []
    for idx, b in enumerate(dataloader1):
        encoder1_feature, mask_TCR = UniTCR_example.encoder_TCR.eval()(b[0].to(device))
        encoder3_feature, mask_pMHC = UniTCR_example.encoder_pMHC.eval()(b[1].to(device))
        pMHC_TCR_embedding = UniTCR_example.fusion_encoder.eval()(encoder3_feature, encoder1_feature, mask_TCR)
        pMHC_TCR_embedding = UniTCR_example.pooler.eval()(pMHC_TCR_embedding, mask_pMHC)
        hid_states = pMHC_TCR_embedding      
        for idx_head, h in enumerate(UniTCR_example.ptm_head):
            hid_states = h.eval()(hid_states)
        logits_b = hid_states
        
        scores = F.softmax(logits_b, dim = 1)

        rank_output += list(scores[:, 1].cpu().numpy())
    rank_output = np.array(rank_output)

# construct the output result dataframe
rank_output_df = pd.DataFrame({'Beta':testing_data['Beta'], 
                            #    'HLA':testing_data['HLA'], 
                            'Peptide':testing_data['Peptide'],
                            'Label':testing_data['Label'], 
                            'Rank':rank_output}
                            )

# store the output result
rank_output_df.to_csv(f"{config['Train']['output_dir']}/Prediction_result.csv", index = False)
print(f"Prediction Accomplished.\n")