import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import joblib
import scanpy as sc
import time


class InputDataSet(Dataset):
    
    def __init__(self, profile_dir, compressed_profile_dir, compressed_TCR_dir):
        # obtain the profile data and Tcell information
        data = sc.read_h5ad(profile_dir)
        
        compressed_data = sc.read_h5ad(compressed_profile_dir)
        
        compressed_TCR = sc.read_h5ad(compressed_TCR_dir)
        
        # obtain the single cell profile data
        self.profile = data.X
        
        self.compressed_profile = compressed_data.X
        
        self.compressed_TCR = compressed_TCR.X
        
        # obtain the profile gene name
        self.gene_names = list(data.var.index)   

        self.beta_chains = list(data.obs['beta'])
        n = 0
        self.TCR_ids = {}
        for idx, beta in enumerate(self.beta_chains):
            if beta not in self.TCR_ids:
                self.TCR_ids[beta] = n
                n += 1
        
    def __getitem__(self, item):
        batch_profile = self.profile[item]
        batch_compressed_profile = self.compressed_profile[item]
        batch_compressed_TCR = self.compressed_TCR[item]
        
        return batch_profile, batch_compressed_profile, batch_compressed_TCR, self.TCR_ids[self.beta_chains[item]]
        
    def __len__(self, ):
        
        # return the cell number of the dataset
        return self.profile.shape[0]
