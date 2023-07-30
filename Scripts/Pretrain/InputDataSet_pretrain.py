import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import joblib
import scanpy as sc
import time

class InputDataSet(Dataset):
    
    def __init__(self, data_dir):
        # obtain the profile data and Tcell information
        data = sc.read_h5ad(data_dir)
        
        # obtain the single cell profile data
        self.profile = data.layers['scale_data'].toarray()
        
        # obtain the profile gene name
        self.gene_names = list(data.var.index)
        
        # obtain the T cell information
        self.beta_chains = list(data.obs['beta'])
             
        # create the position encoding
        self.position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(self.position_encoding)
        self.aa_dict = joblib.load("/home/gaoyicheng/pep_tcr_with_gyl/TCRBagger/Requirements/dic_Atchley_factors.pkl")
        
        # create the TCR index dict
        # beta
        n = 0
        self.TCR_ids = {}
        for idx, beta in enumerate(self.beta_chains):
            if beta not in self.TCR_ids:
                self.TCR_ids[beta] = n
                n += 1
        
    def aamapping(self,TCRSeq,encode_dim):
        #the longest sting will probably be shorter than 80 nt
        TCRArray = []
        if len(TCRSeq)>encode_dim:
            # print('Length: '+str(len(TCRSeq))+' over bound!')
            TCRSeq=TCRSeq[0:encode_dim]
        for aa_single in TCRSeq:
            try:
                TCRArray.append(self.aa_dict[aa_single])
            except KeyError:
                # print('Not proper aaSeqs: '+TCRSeq)
                TCRArray.append(np.zeros(5,dtype='float64'))
        for i in range(0,encode_dim-len(TCRSeq)):
            TCRArray.append(np.zeros(5,dtype='float64'))
        return torch.FloatTensor(np.array(TCRArray)) 

    def add_position_encoding(self,seq):
        mask = (seq == 0).all(dim = 1)
        seq[~mask] += self.position_encoding[:seq[~mask].size()[-2]]
        return seq 
    
    def __getitem__(self, item):
        batch_profile = self.profile[item]
        TCR_beta = self.add_position_encoding(self.aamapping(self.beta_chains[item], 25))
        
        return TCR_beta, batch_profile, self.TCR_ids[self.beta_chains[item]], item
        
    def __len__(self, ):
        
        # return the cell number of the dataset
        return self.profile.shape[0]
