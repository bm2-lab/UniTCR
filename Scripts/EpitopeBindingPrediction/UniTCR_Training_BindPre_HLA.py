"""
training command line:
python ./Scripts/EpitopeBindingPrediction/UniTCR_Training_BindPre_HLA.py  --config ./Configs/TrainingConfig_EpitopeBindPrediction_HLA.yaml
"""
import scanpy as sc
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import joblib
import pandas as pd
from InputDataSet_Training_HLA import InputDataSet1
from InputDataSet_Testing_HLA import InputDataSet_val
from torch.nn import functional as F
from Modelblocks_UniTCR_BindPre import UniTCR_model

import logging
import yaml
from sklearn.metrics import roc_auc_score
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-C","--config", required = True, help = "path the config file")
args = ap.parse_args()

class EarlyStopping:
    def __init__(self, patience = 20, verbose = False, delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.earlystopping = False
        self.best_score = None
        
    def __call__(self, val_auc):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.earlystopping = True
        else:
            self.best_score = score
            self.counter = 0

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
# config_file = "./TrainingConfig.yaml"
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
UniTCR_example = UniTCR_model(encoderTCR_in_dim = config['Train']['Model_Parameter']['encoderTCR_in_dim'],
                              encoderTCR_out_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                              encoderpMHC_in_dim = config['Train']['Model_Parameter']['encoderpMHC_in_dim'],
                              encoderpMHC_out_dim = config['Train']['Model_Parameter']['encoderpMHC_out_dim'],
                              out_dim = config['Train']['Model_Parameter']['out_dim'],
                              head_num = config['Train']['Model_Parameter']['head_num'],
                              ).to(device)

if config['Train']['Model_Parameter']['Pretrained']:
    pretrained_parts = ['encoder_TCR', 'TCR_proj', 'pooler']
    pretrained_dict = torch.load("./Requirements/Pretrained_model.pth")['model_state_dict']
    model_dict_tcr_encoder = UniTCR_example.encoder_TCR.state_dict()
    pretrained_dict_tmp = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if '.'.join(k.split('.')[1:]) in model_dict_tcr_encoder}
    model_dict_tcr_encoder.update(pretrained_dict_tmp)
    UniTCR_example.encoder_TCR.load_state_dict(model_dict_tcr_encoder)
    # for param in UniTCR_example.encoder_TCR.parameters():
    #     param.requires_grad = False
        
    # model_dict_pooler = UniTCR_example.pooler.state_dict()
    # pretrained_dict_tmp = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if '.'.join(k.split('.')[1:]) in model_dict_pooler}
    # model_dict_pooler.update(pretrained_dict_tmp)
    # UniTCR_example.pooler.load_state_dict(model_dict_pooler)

# for param in UniTCR_example.encoder_TCR.parameters():
#     param.requires_grad = config['Train']['Model_Parameter']['Pretrained']
    
# setting the learning rate for the model
non_frozen_parameters = [p for p in UniTCR_example.parameters() if p.requires_grad]
UniTCR_example.optimizer = torch.optim.AdamW(non_frozen_parameters, lr = config['Train']['Trainer_parameter']['learning_rate'])

# load checkpoint
# checkpoint = torch.load("/home/gaoyicheng/BM2_Projects/DalleTCR/IncompleteTripleModality/BaselineModelWithContrastive/Model_checkpoints/epoch1010_model.pth")
# DalleTCR_example.load_state_dict(checkpoint['model_state_dict'])
# DalleTCR_example.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

# initialize the dataset for the batch training
dataset1 = InputDataSet1(config['dataset']['Training_dataset'])
dataset_val = InputDataSet_val(config['dataset']['Validation_dataset'])

# initialize the dataloader
dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size = config['Train']['Sampling']['batch_size'], shuffle = config['Train']['Sampling']['sample_shuffle'])
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = 10000, shuffle = False)

# initialize the logger for saving the traininig log
logger = get_logger(config['Train']['output_dir']+'/training.log')

# setting the training epoch
epochs = config['Train']['Trainer_parameter']['epoch']

# setting earlystopping
early_stopping = EarlyStopping()

# training the UniTCR
for epoch in range(1, epochs + 1):
    loss_task1 = 0
    # training UniTCR with task_1
    s = time.time()
    UniTCR_example.train()
    for idx, b in enumerate(dataloader1):
        # only beta
        loss_b = UniTCR_example([b[0].to(device), 
                                            b[1].to(device)], 
                                            b[2].to(device)
                                            )
        UniTCR_example.optimizer.zero_grad()
        loss_b.backward()
        UniTCR_example.optimizer.step()
        loss_task1 += loss_b.item()
    e = time.time()
    loss_task1 /= (idx + 1)
    
    UniTCR_example.eval()
    with torch.no_grad():
        rank_output = []
        labels = []
        for idx_val, b_val in enumerate(dataloader_val):
            encoder1_feature, mask_TCR = UniTCR_example.encoder_TCR.eval()(b_val[0].to(device))
            encoder3_feature, mask_pMHC = UniTCR_example.encoder_pMHC.eval()(b_val[1].to(device))
            pMHC_TCR_embedding = UniTCR_example.fusion_encoder.eval()(encoder3_feature, encoder1_feature, mask_TCR)
            pMHC_TCR_embedding = UniTCR_example.pooler.eval()(pMHC_TCR_embedding, mask_pMHC)
            
            hid_states = pMHC_TCR_embedding      
            for idx_head, h in enumerate(UniTCR_example.ptm_head):
                hid_states = h.eval()(hid_states)
            logits_b = hid_states
            scores = F.softmax(logits_b, dim = 1)
            rank_output += list(scores[:, 1].cpu().numpy())
            labels += list(b_val[2].numpy())
        rank_output = np.array(rank_output)
        labels = np.array(labels)
        ROC_auc_val = roc_auc_score(labels, rank_output)
        loss_vals = nn.BCELoss()(torch.Tensor(rank_output),torch.Tensor(labels).float())
    logger.info('Epoch:[{}/{}]\tsteps:{}\tloss:{:.5f}\tval_loss:{:.5f}\tval_AUC:{:.5f}\ttime:{:.3f}'.format(epoch, epochs, idx + 1, loss_task1, loss_vals, ROC_auc_val, e-s))
    
    # checkpoint saving
    if epoch % 10 == 0:
        torch.save({'epoch':epoch, 
                    'model_state_dict':UniTCR_example.state_dict(), 
                    'optimizer_state_dict': UniTCR_example.optimizer.state_dict(), 
                    'loss': loss_task1 / (idx+1)}, 
                f"{config['Train']['output_dir']}/Model_checkpoints/epoch{epoch}_model.pth")
        
    # early_stopping(ROC_auc_val)
    # if early_stopping.earlystopping:
    #     print("Early stopping")
    #     break
    
logger.info('finish training!')