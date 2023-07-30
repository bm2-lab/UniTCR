import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import joblib
import scanpy as sc
import time


class InputDataSet_val(Dataset):
    
    def __init__(self, data_dir):
        # loading the information of pMHC-TCR pairs
        if type(data_dir) == str:
            data = pd.read_csv(data_dir)
        else:
            data = data_dir
        
        # obtain the alpha and beta chain information
        # self.alpha_chains = data['Alpha']
        self.beta_chains = data['Beta']
        
        # obtain the peptide and MHC information
        self.peptides = data['Peptide']
     
        self.labels = data['Label']
          
        # create the position encoding
        self.position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(self.position_encoding)
        self.aa_dict = joblib.load("./Requirements/dic_Atchley_factors.pkl")
        
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
        TCR_beta = self.add_position_encoding(self.aamapping(self.beta_chains[item], 25))
        pMHC_embeddings = self.add_position_encoding(self.aamapping(self.peptides[item], 25))
        return TCR_beta, pMHC_embeddings, self.labels[item]
    def __len__(self, ):
        
        # return the cell number of the dataset
        return self.beta_chains.shape[0]

# def __main__():
# dataset = InputDataSet1("/home/gaoyicheng/BM2_Projects/DalleTCR/Database/Integrated_TCR_Peptide_Paired.csv")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)