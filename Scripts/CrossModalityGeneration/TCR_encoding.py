import pandas as pd
import joblib
import numpy as np
import torch

# create the position encoding
position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
position_encoding = torch.from_numpy(position_encoding)
aa_dict = joblib.load("./Requirements/dic_Atchley_factors.pkl")

def aamapping(TCRSeq,encode_dim):
    TCRArray = []
    if len(TCRSeq)>encode_dim:
        # print('Length: '+str(len(TCRSeq))+' over bound!')
        TCRSeq=TCRSeq[0:encode_dim]
    for aa_single in TCRSeq:
        try:
            TCRArray.append(aa_dict[aa_single])
        except KeyError:
            # print('Not proper aaSeqs: '+TCRSeq)
            TCRArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,encode_dim-len(TCRSeq)):
        TCRArray.append(np.zeros(5,dtype='float64'))
    return torch.FloatTensor(np.array(TCRArray)) 

def add_position_encoding(seq):
    mask = (seq == 0).all(dim = 1)
    seq[~mask] += position_encoding[:seq[~mask].size()[-2]]
    return seq 

def TCR_encoding(beta_chain):
    # TCR_alpha = add_position_encoding(aamapping(alpha_chain, 25))
    TCR_beta = add_position_encoding(aamapping(beta_chain, 25))
    return TCR_beta