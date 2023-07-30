import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.nn import functional as F

class Input_embedding(nn.Module):
    """
    this is the class used for embedding input
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
        
    Returns:
        a new normalized data(bs, seq_len, out_dim) based on the linear transformation and layer normalization
    """
    
    def __init__(self, in_dim = 5, out_dim = 5):
        super().__init__()
        
        # define the dimension expansion layer, i.e. mapping original dimension into a new space
        self.expand = nn.Linear(in_dim, out_dim)
        
        # define the layer normalization
        self.layer_norm = nn.LayerNorm(out_dim)
        
        # define dropout layer
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, input_data):
        
        # expand the input data
        expand_data = self.expand(input_data)
        
        # layer normalization the input data
        normalized_data = self.layer_norm(expand_data)
        
        # dropout
        normalized_data = self.dropout(normalized_data)
        
        return normalized_data
    
class Self_attention(nn.Module):
    """
    this is the basic self-attention class
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        hidden states(bs, seq_len, out_dim) based on the self-attention mechanism 
    """
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1):
        super().__init__()
        if out_dim % head_num != 0:
            raise ValueError(
                "Hidden size is not a multiple of the number of attention head"
            )
        
        # define the head number
        self.head_num = head_num
        
        # define the head size
        self.head_size = int(out_dim / head_num)
        
        # define the all head dimension
        self.out_dim = out_dim
        
        # define the Q, K, V matrix
        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
    
    def transpose_for_score(self, x):
        new_x_shape = x.size()[:-1] + (self.head_num, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, input_data, mask):
        
        # calculate the Q, K, V matrix
        Q = self.Q(input_data)
        K = self.K(input_data)
        V = self.V(input_data)
        
        # multi-head
        Q = self.transpose_for_score(Q)
        K = self.transpose_for_score(K)
        V = self.transpose_for_score(V)
        
        # calculate the logits based on the Query matrix and key matrix
        score = torch.matmul(Q, K.transpose(-1, -2))
        score /= math.sqrt(self.head_size)
        
        # mask the padding
        # score.transpose(-1, -2)[mask] = -1e15
        new_mask = mask[:, None, None, :] * (-1e15)
        score = score + new_mask
        
        # calculate the attention score
        att = F.softmax(score, dim = -1)
        
        # output the result based on the attention and value
        output = torch.matmul(att, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        new_output_shape = output.size()[:-2] + (self.out_dim, )
        output = output.view(*new_output_shape)
        
        return output

class Attention_output(nn.Module):
    """
    this is the linear transformation class
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        hidden states(bs, seq_len, out_dim) based on the feedforward and residual connection
    """
    
    def __init__(self, in_dim = 5, out_dim = 5):
        super().__init__()
        
        # define the linear layer
        self.feedforward = nn.Linear(in_dim, out_dim)
        
        # define the layer normalization layer
        self.layernorm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, hidden_states, input_data):
        
        # feedforward the input data
        hidden_states = self.feedforward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # resnet the input data and output with layer normalization
        hidden_states = self.layernorm(hidden_states + input_data)
        
        return hidden_states

class Attention_intermediate(nn.Module):
    """
    this is the intermediate class
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
        
    Returns:
        hidden states(bs, seq_len, out_dim) base on the feedforward and gelu activattion
    """
    
    def __init__(self, in_dim = 5, out_dim = 5):
        super().__init__()
        
        # define the linear layer
        self.feedforward = nn.Linear(in_dim, out_dim)
        
        # define the gelu as activate function
        self.gelu = nn.GELU()
        
    def forward(self, input_data):
        
        # feedforward the input data
        hidden_states = self.feedforward(input_data)
        
        # activate the hidden state
        hidden_states = self.gelu(hidden_states)
        
        return hidden_states
    
class Attention_pooler(nn.Module):
    """
    this is the mean pooling class
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        hidden state(bs, out_dim) output by masking the padding and averaging the seq
    """
    
    def __init__(self, in_dim = 5, out_dim = 5):
        super().__init__()
        
        # define the linear layer
        self.feedforward = nn.Linear(in_dim, out_dim)
        
        # define the tanh as activation function
        self.activation = nn.Tanh()
    def forward(self, input_data, mask):
        
        # setting the value of padding as 0
        input_data[mask] = 0
        
        # calculate the mean value
        pooled_output = input_data.sum(dim = 1) * (1 / (~mask).sum(dim = 1)).unsqueeze(0).T
        
        # linear transformation of output
        pooled_output = self.feedforward(pooled_output)
        
        # activate the output
        pooled_output = self.activation(pooled_output)
        
        return pooled_output

class Self_attention_layer(nn.Module):
    """
    this is the self_attention_layer class, which contains self-attention blocks and output layer with residual connection
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        hidden states(bs, seq_len, out_dim)
    """
    
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1):
        super().__init__()
        
        # define the self-attention layer
        self.self_attention = Self_attention(in_dim, out_dim, head_num)
        
        # define the attention-output layer
        self.self_output = Attention_output(out_dim, in_dim)
        
    def forward(self, input_data, mask):
        
        # input data into the self-attention layer
        hidden_states = self.self_attention(input_data, mask)
        
        # feedforward the output
        hidden_states = self.self_output(hidden_states, input_data)
        
        return hidden_states

class Self_attention_encoder_layer(nn.Module):
    """
    this is the self-attention encoder blocks, which contains self-attention-layer, intermediate and output layer
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
        
    Returns:
        hiddens states(bs, seq_len, out_dim)
    """
    
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1):
        super().__init__()
        
        # define the self-attention-layer
        self.attention_layer = Self_attention_layer(in_dim, out_dim, head_num)
        
        # define the attention intermediate layer
        self.intermediate = Attention_intermediate(out_dim, out_dim)
        
        # define the attention-output layer
        self.output = Attention_output(out_dim, in_dim)
    
    def forward(self, input_data, mask):
        
        # self-attention-layer output
        hidden_states = self.attention_layer(input_data, mask)
        
        # output of intermediate
        intermediate_states = self.intermediate(hidden_states)
        
        # output the feedforward layer
        output = self.output(intermediate_states, hidden_states)
        
        return output

class Cross_attention(nn.Module):
    """
    this is the cross-attention class
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hiddens states(bs, query_seq_len, out_dim)
    """
    
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1):
        super().__init__()
        
        if out_dim % head_num != 0:
            raise ValueError(
                "Hidden size is not a multiple of the number of attention head"
            )
        
        # define the head number
        self.head_num = head_num
        
        # define the head size
        self.head_size = int(out_dim / head_num)
        
        # define the all head dimension
        self.out_dim = out_dim        
        
        # define the Q, K, V matrix
        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
    
    def transpose_for_score(self, x):
        new_x_shape = x.size()[:-1] + (self.head_num, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query_input, value_input, v_mask):
        # size:(btz, n_samples, seq_len, features)
        
        # calculate the Q, K, V matrix
        Q = self.Q(query_input)
        K = self.K(value_input)
        V = self.V(value_input)
        
        # Multi-head
        Q = self.transpose_for_score(Q)
        K = self.transpose_for_score(K)
        V = self.transpose_for_score(V)
        
        # calculate the logits of cross-attention
        score = torch.matmul(Q, K.transpose(-1, -2))
        score /= math.sqrt(self.head_size)
        
        # mask the padding
        # score.transpose(1, 2)[v_mask] = -1e15
        new_mask = v_mask[:, None, None, :] * (-1e15)
        score = score + new_mask
        
        # calculate the attention score
        att = F.softmax(score, dim = -1)
        
        # output the result based on the attention and value
        output = torch.matmul(att, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        new_output_shape = output.size()[:-2] + (self.out_dim, )
        output = output.view(*new_output_shape)        
        
        return output

class Cross_attention_layer(nn.Module):
    """
    this is the cross-attention-layer class, which contains cross-attention and output layer
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        hidden states(bs, query_seq_len, out_dim)
    """
    
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1):
        super().__init__()
        
        # define the cross-attention layer
        self.cross_attention = Cross_attention(in_dim, out_dim, head_num)
        
        # define the feedforward layer
        self.self_output = Attention_output(out_dim, in_dim)
        
    def forward(self, query_input, value_input, v_mask):
        
        # calculate the cross-attention
        hidden_states = self.cross_attention(query_input, value_input, v_mask)
        
        # calculate the feedforward output
        hidden_states = self.self_output(hidden_states, query_input)
        
        return hidden_states

class Cross_attention_encoder_layer(nn.Module):
    """
    this is the cross-attention-encoder-layer, which constains self-attention-layer, cross-attention-layer
    intermediate layer and output layer
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        hidden states(bs, query_seq_len, out_dim)
    """
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1):
        super().__init__()
        
        # define the self-attention-layer
        self.self_attention_layer = Self_attention_layer(in_dim, out_dim, head_num)
        
        # define the cross-attention-layer
        self.cross_attention_layer = Cross_attention_layer(in_dim, out_dim, head_num)
        
        # define the intermediate layer
        self.intermediate = Attention_intermediate(in_dim, out_dim)
        
        # define the output layer
        self.output = Attention_output(out_dim, in_dim)
    
    def forward(self, query_input, value_input, v_mask):
        
        # output the result of self-attention-layer based on the value input
        hidden_value_states = self.self_attention_layer(value_input, v_mask)
        
        # output the result of cross-attention-layer
        hidden_states = self.cross_attention_layer(query_input, value_input, v_mask)
        
        # output the result of intermediate
        intermediate_states = self.intermediate(hidden_states)
        
        # output the feedforward
        output = self.output(intermediate_states, hidden_states)
        
        return output
        
class Encoder_TCR(nn.Module):
    """
    this is the TCR encoder class, which contains the self-attentioin-layer of alpha chain,
    the self-attention-layer of beta chain and the cross-attention-layer of alpha and beta chain
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        TCR embedding(bs, query_seq_len, out_dim)
        alpha chain padding mask
    """
    
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1, initial_dim = 5):
        super().__init__()
        
        # define Input embedding
        self.input_embedding_TCR = Input_embedding(initial_dim, out_dim)
        
        # self.alpha_encoder = Self_attention_encoder_layer(in_dim, out_dim, head_num)
        self.beta_encoder = Self_attention_encoder_layer(in_dim, out_dim, head_num)
        # self.alpha_beta_fusion_encoder = Cross_attention_encoder_layer(in_dim, out_dim, head_num)
        

        
    def forward(self, TCR_beta_encoding, TCR_alpha_encoding = None):
        '''
        alpha_mask = (TCR_alpha_encoding == 0).all(dim = 2)
        beta_mask = (TCR_beta_encoding == 0).all(dim = 2)
        TCR_alpha_encoding = self.alpha_encoder(TCR_alpha_encoding, alpha_mask)
        TCR_beta_encoding = self.beta_encoder(TCR_beta_encoding, beta_mask)
        TCR_embedding = self.alpha_beta_fusion_encoder(TCR_alpha_encoding, TCR_beta_encoding, beta_mask)
        return (TCR_embedding, alpha_mask,)
        '''
        # beta only
        beta_mask = (TCR_beta_encoding == 0).all(dim = 2)
        TCR_beta_encoding = self.input_embedding_TCR(TCR_beta_encoding)
        TCR_beta_encoding = self.beta_encoder(TCR_beta_encoding, beta_mask)
        
        return (TCR_beta_encoding, beta_mask,)    
        
class Encoder_pMHC(nn.Module):
    """
    this is the pMHC encoder class, which contains the self-attentioin-layer of HLA,
    the self-attention-layer of peptide and the cross-attention-layer of HLA and peptide
    
    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension
    
    Returns:
        pMHC embedding(bs, query_seq_len, out_dim)
        MHC padding mask
    """
    
    def __init__(self, in_dim = 5, out_dim = 5, head_num = 1, initial_dim = 5):
        super().__init__()
        
        # define Input embedding
        self.input_embedding_pMHC = Input_embedding(initial_dim, out_dim)
        # self.hla_encoder = Self_attention_encoder_layer(in_dim, out_dim, head_num)
        self.peptide_encoder = Self_attention_encoder_layer(in_dim, out_dim, head_num)
        # self.hla_peptide_fusion_encoder = Cross_attention_encoder_layer(in_dim, out_dim, head_num)
    
    def forward(self, peptide_encoding):
        
        # obtain the initial peptide encoding
        peptide_mask = (peptide_encoding == 0).all(dim = 2)
        
        peptide_encoding = self.input_embedding_pMHC(peptide_encoding)
        peptide_encoding = self.peptide_encoder(peptide_encoding, peptide_mask)
        
        return (peptide_encoding, peptide_mask,)
        
class Encoder_pMHC_Bilstm(nn.Module):
    def __init__(self, in_dim = 5, out_dim = 64):
        super().__init__()
        self.bilstm_encoder = nn.LSTM(in_dim, out_dim, 2, batch_first = True, bidirectional = False, dropout = 0)
        self.layer_norm = nn.LayerNorm(out_dim)
    def forward(self, peptide_encoding):
        peptide_mask = (peptide_encoding == 0).all(dim = 2)
        peptide_encoding, (hn,cn) = self.bilstm_encoder(peptide_encoding)
        peptide_encoding = self.layer_norm(peptide_encoding)
        return (peptide_encoding, peptide_mask,)
    
# define the projection head for mapping each modality into a same dimension
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout = 0.2):
        super().__init__()
        
        # define the projection linear layer
        self.projection = nn.Linear(in_dim, out_dim)
        
        # define the gelu activation function
        self.gelu = nn.GELU()
        
        # define the linear mapping function
        self.fc = nn.Linear(out_dim, out_dim)
        
        # define the dropout
        self.dropout =  nn.Dropout(dropout)
        
        # define the layer normalization layer
        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine = False)
        
    def forward(self, x):
        
        # various dimention mapping
        h = self.projection(x)
        
        # feature activation
        x = self.gelu(h)
        
        # linear mapping linear
        x = self.fc(x)
        
        # dropout layer
        x = self.dropout(x)
        
        # residue connection
        x = x + h
        
        # layer normalization
        x = self.layer_norm(x)
        
        return x

class UniTCR_model(nn.Module):
    def __init__(self, 
                 encoderTCR_in_dim, encoderTCR_out_dim,
                 encoderpMHC_in_dim, encoderpMHC_out_dim, 
                 out_dim, 
                 head_num,
                 temperature = 0.2, 
                 learning_rate = 0.001
                 ):
        super().__init__()

        # define the encoder
        self.encoder_TCR = Encoder_TCR(encoderTCR_out_dim, encoderTCR_out_dim, head_num)

        # self.encoder_pMHC = Encoder_pMHC(encoderpMHC_out_dim, encoderpMHC_out_dim, head_num)
        self.encoder_pMHC = Encoder_pMHC_Bilstm(encoderpMHC_in_dim, encoderpMHC_out_dim)
        
        # define the cross-attention layer
        self.fusion_encoder = Cross_attention_encoder_layer(encoderpMHC_out_dim, encoderTCR_out_dim, head_num)
        
        # define the pooling layer
        self.pooler = Attention_pooler(out_dim, out_dim)
        
        # define the final head layer
        # self.ptm_head = nn.Linear(out_dim, 2)
        self.ptm_head = nn.ModuleList([nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(0.5), 
                                nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.5) , 
                                nn.Linear(64, 2, bias = False)])

        # define the optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)   
    
    def forward(self, x, BG_TCRs, task_level=None):
        
        # extract the features of TCR
        encoderTCR_feature, mask_TCR = self.encoder_TCR(x[0])
        
        # extract the pMHC features based on the encoder3
        encoderpMHC_feature, mask_pMHC = self.encoder_pMHC(x[1])
        
        # using mean pool as the sequence embedding
        encoderpMHC_embedding = encoderpMHC_feature.sum(dim = 1) * (1 / (~mask_pMHC).sum(dim = 1)).unsqueeze(0).T

         # extract the features of Bachground TCR
        encoderTCR_feature_bg, mask_TCR_bg = self.encoder_TCR(BG_TCRs)
        
        ############# PTM ##############
        bs = encoderTCR_feature.size()[0]
        
        # output the fusion embedding of positive pairs
        pMHC_TCR_embedding = self.fusion_encoder(encoderpMHC_feature, encoderTCR_feature, mask_TCR)
        pMHC_TCR_embedding = self.pooler(pMHC_TCR_embedding, mask_pMHC)
        
        pMHC_TCR_embedding_bg = self.fusion_encoder(encoderpMHC_feature, encoderTCR_feature_bg, mask_TCR_bg)
        pMHC_TCR_embedding_bg = self.pooler(pMHC_TCR_embedding_bg, mask_pMHC)
        
        # calculate the logits
        hid_states = torch.cat([pMHC_TCR_embedding, pMHC_TCR_embedding_bg], dim = 0)
        for idx, h in enumerate(self.ptm_head):
            hid_states = h(hid_states)
        logits = hid_states
        targets = torch.cat([torch.ones(bs, dtype = torch.long), torch.zeros(bs, dtype = torch.long)], dim = 0).to(encoderTCR_feature.device)
        # calculate the ptm loss
        loss_ptm = F.cross_entropy(logits, targets)
        
        return loss_ptm

def __main__():
    x_1 = torch.rand(10,100)
    x_2 = torch.rand(10,200)
    x_3 = torch.rand(10,50,100)
    U_example = torch.rand(10,50)
    x = [x_1, x_2, x_3]
    CLIP_example = CLIP_model(100, 50, 200, 150, 100, 50, 32)

    for epoch in range(2000):
        loss_b = CLIP_example(x, U_example)
        CLIP_example.optimizer.zero_grad()
        loss_b.backward()
        CLIP_example.optimizer.step()
        print(f"epoch:{epoch}, loss:{loss_b}")
