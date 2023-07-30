"""
training command line:
python ./Scripts/CrossModalityGeneration/UniTCR_training_CrossModalityGeneration.py --config ./Configs/TrainingConfig_CrossModalityGeneration.yaml
"""
import scanpy as sc
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
from InputDataSet_CrossModalityGeneration import InputDataSet
from torch.nn import functional as F
from UniTCR_Modelblocks_CrossModalityGeneration import Decoder_model
import logging
import yaml
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-C","--config", required = True, help = "path the config file")
args = ap.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

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

# create directory if it not exist
if not os.path.exists(config['Train']['output_dir']):
    os.makedirs(config['Train']['output_dir'])
    os.makedirs(config['Train']['output_dir']+'/Model_checkpoints')

# initialize a new UniTCR model
Decoder_model_example = Decoder_model(compressed_TCR_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                                      compressed_profile_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'],
                                      decoderprofile_in_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'], 
                                      decoderprofile_hid_dim = config['Train']['Model_Parameter']['encoderprofile_hid_dim'], 
                                      decoderprofile_hid_dim2 = config['Train']['Model_Parameter']['encoderprofile_hid_dim2'],
                                      decoderprofile_out_dim = config['Train']['Model_Parameter']['encoderprofle_out_dim'],
                                      ).to(device)

# setting the learning rate for the model
Decoder_model_example.optimizer = torch.optim.AdamW(Decoder_model_example.parameters(), lr = config['Train']['Trainer_parameter']['learning_rate'])

# initialize the dataset for the batch training
dataset = InputDataSet(config['dataset']['Training_dataset'], config['dataset']['Training_compressed_profile'], config['dataset']['Training_compressed_TCR'])
dataset_val = InputDataSet(config['dataset']['Validation_dataset'], config['dataset']['Validation_compressed_profile'], config['dataset']['Validation_compressed_TCR'])

# initialize the dataloader
dataloader1 = torch.utils.data.DataLoader(dataset, batch_size = config['Train']['Sampling']['batch_size'], shuffle = config['Train']['Sampling']['sample_shuffle'])
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = 64, shuffle = False)
# initialize the logger for saving the traininig log
logger = get_logger(config['Train']['output_dir']+'/training.log')

# setting the training epoch
epochs = config['Train']['Trainer_parameter']['epoch']

# training the UniTCR
for epoch in range(1, epochs + 1):
    loss_task_prior = 0
    loss_task_de = 0
    loss_task_kl = 0
    # training UniTCR
    s = time.time()
    for idx, b in enumerate(dataloader1):
        loss_b_prior, loss_b_de, loss_b_kl = Decoder_model_example(b[1].to(device), b[0].to(device), b[2].to(device), b[3].to(device))
        # loss_b = 0 * loss_b_prior + 0.5 * loss_b_de + 0.5 * loss_b_kl
        
        if epoch < 70:
            loss_b = 1 * loss_b_prior + 0 * loss_b_de + 0 * loss_b_kl
        elif epoch == 70: 
            non_frozen_parameters = [p for p in Decoder_model_example.decoder_profile.parameters()]
            Decoder_model_example.optimizer = torch.optim.AdamW(non_frozen_parameters, lr = config['Train']['Trainer_parameter']['learning_rate'])
            loss_b = 0 * loss_b_prior + 0.5 * loss_b_de + 0.5 * loss_b_kl
        else:
            loss_b = 0 * loss_b_prior + 0.5 * loss_b_de + 0.5 * loss_b_kl

        Decoder_model_example.optimizer.zero_grad()
        loss_b.backward()
        Decoder_model_example.optimizer.step()
        loss_task_prior += loss_b_prior.item()
        loss_task_de += loss_b_de.item()
        loss_task_kl += loss_b_kl.item()
        
    e = time.time()
    loss_task_prior /= idx + 1
    loss_task_de /= idx + 1
    loss_task_kl /= idx + 1
    
    original_profile_val = []
    original_compressed_profile_val = []
    compressed_profile_val = []
    decompressed_profile_val = []
    loss_b_prior_val = 0
    loss_b_de_val = 0
    for idx_val, b_val in enumerate(dataloader_val):
        with torch.no_grad():
            compressed_profile = Decoder_model_example.TCR2profile.eval()(b_val[2].to(device))
            decompressed_profile = Decoder_model_example.decoder_profile.eval()(compressed_profile)
            original_profile_val.append(b_val[0])
            original_compressed_profile_val.append(b_val[1].cpu())
            compressed_profile_val.append(compressed_profile.cpu())
            decompressed_profile_val.append(decompressed_profile.cpu())

            loss_b_prior_val_batch, loss_b_de_val_batch, loss_b_kl_val_batch = Decoder_model_example.eval()(b_val[1].to(device), b_val[0].to(device), b_val[2].to(device), b_val[3].to(device))
            loss_b_prior_val += loss_b_prior_val_batch
            loss_b_de_val += loss_b_de_val_batch
            
    original_profile_val = torch.cat(original_profile_val, dim = 0)
    original_compressed_profile_val = torch.cat(original_compressed_profile_val, dim = 0)
    compressed_profile_val = torch.cat(compressed_profile_val, dim = 0)
    decompressed_profile_val = torch.cat(decompressed_profile_val, dim = 0)

    loss_b_prior_val /= idx_val + 1
    loss_b_de_val /= idx_val + 1
    
    logger.info('Epoch:[{}/{}]\tsteps:{}\tloss_prior:{:.5f}\tloss_de:{:.5f}\tloss_kl:{:.5f}\tloss_prior_val:{:.5f}\tloss_de_val:{:.5f}\ttime:{:.3f}'.format(epoch, epochs, idx + 1, 
                                                                                                                                          loss_task_prior, loss_task_de, loss_task_kl, 
                                                                                                                                          loss_b_prior_val, loss_b_de_val, 
                                                                                                                                          e-s))
    
    # checkpoint saving
    if epoch % 10 == 0:
        torch.save({'epoch':epoch, 
                    'model_state_dict':Decoder_model_example.state_dict(), 
                    'optimizer_state_dict': Decoder_model_example.optimizer.state_dict()
                    }, 
                f"{config['Train']['output_dir']}/Model_checkpoints/epoch{epoch}_model.pth")
    
logger.info('finish training!')

# saving the trained model
# torch.save(DalleTCR_example.state_dict(), config['Train']['output_dir']+"/DalleTCR_model.pth")