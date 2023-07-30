"""
training command line:
python ./Scripts/Pretrain/UniTCR_pretrain.py --config ./Configs/TrainingConfig_pretrain.yaml
"""

import scanpy as sc
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import torch
from torch import nn
import numpy as np
import math
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

# initialize a new DalleTCR model
UniTCR_example = UniTCR_model(encoderTCR_in_dim = config['Train']['Model_Parameter']['encoderTCR_in_dim'],
                              encoderTCR_out_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                              encoderprofile_in_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'], 
                              encoderprofile_hid_dim = config['Train']['Model_Parameter']['encoderprofile_hid_dim'], 
                              encoderprofile_hid_dim2 = config['Train']['Model_Parameter']['encoderprofile_hid_dim2'],
                              encoderprofle_out_dim = config['Train']['Model_Parameter']['encoderprofle_out_dim'],
                              out_dim = config['Train']['Model_Parameter']['out_dim'],
                              head_num = config['Train']['Model_Parameter']['head_num'],
                              ).to(device)

# setting the learning rate for the model
UniTCR_example.optimizer = torch.optim.AdamW(UniTCR_example.parameters(), lr = config['Train']['Trainer_parameter']['learning_rate'])

dataset = InputDataSet(config['dataset']['Training_dataset'])
data_dist_tcr = sc.read_h5ad(config['dataset']['Training_tcr_dist'])
data_dist_tcr = torch.Tensor(data_dist_tcr.X)
data_dist = sc.read_h5ad(config['dataset']['Training_profile_dist'])
data_dist = torch.Tensor(data_dist.X)

# initialize the dataloader
dataloader1 = torch.utils.data.DataLoader(dataset, batch_size = config['Train']['Sampling']['batch_size'], shuffle = config['Train']['Sampling']['sample_shuffle'])

# create directory if it not exist
if not os.path.exists(config['Train']['output_dir']):
    os.makedirs(config['Train']['output_dir'])
    os.makedirs(config['Train']['output_dir']+'/Model_checkpoints')

# initialize the logger for saving the traininig log
logger = get_logger(config['Train']['output_dir']+'/training.log')

# setting the training epoch
epochs = config['Train']['Trainer_parameter']['epoch']

# training the UniTCR
flag = True
for epoch in range(1, epochs + 1):
    loss_ptc_epoch = 0
    loss_ptm_epoch = 0
    loss_p2p_epoch = 0
    loss_t2t_epoch = 0
    loss_task1 = 0
    s = time.time()
    for idx, b in enumerate(dataloader1):   
        similarity_dist_profile = data_dist[b[3]][:,b[3]]
        similarity_dist_tcr = data_dist_tcr[b[3]][:,b[3]]
        loss_ptc, loss_p2p, loss_t2t = UniTCR_example([b[0].to(device), 
                                            b[1].to(torch.float32).to(device)], 
                                            b[2].to(device), 
                                            similarity_dist_profile.to(device), 
                                            similarity_dist_tcr.to(device),
                                            task_level = 1)
        loss_b = 0.1 * loss_ptc + 0.8 * loss_p2p + 0.1 * loss_t2t
        UniTCR_example.optimizer.zero_grad()
        loss_b.backward()
        UniTCR_example.optimizer.step()
        loss_ptc_epoch += loss_ptc.item()
        loss_p2p_epoch += loss_p2p.item()
        loss_t2t_epoch += loss_t2t.item()
        
        loss_task1 += loss_b.item()
    e = time.time()
    loss_ptc_epoch /= (idx + 1)
    loss_p2p_epoch /= (idx + 1)
    loss_t2t_epoch /= (idx + 1)
    logger.info('Epoch:[{}/{}]\tsteps:{}\tptc_loss:{:.5f}\tp2p_loss:{:.5f}\tt2t_loss:{:.5f}\ttime:{:.3f}'.format(epoch, epochs, idx + 1, loss_ptc_epoch, loss_p2p_epoch, loss_t2t_epoch, e-s))
    
    # checkpoint saving
    if epoch % 10 == 0:
        torch.save({'epoch':epoch, 
                    'model_state_dict':UniTCR_example.state_dict(), 
                    'optimizer_state_dict': UniTCR_example.optimizer.state_dict(), 
                    'loss': loss_task1 / (idx+1)}, 
                f"{config['Train']['output_dir']}/Model_checkpoints/epoch{epoch}_model.pth")
    
logger.info('finish training!')