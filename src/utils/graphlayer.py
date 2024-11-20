
from typing import Optional
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    def __init__(self,with_linear=False,emb_size=None,cache_adj=True):
        super(MeanAggregator, self).__init__()
        self.with_linear=with_linear
        if with_linear:
            self.linear=nn.Linear(emb_size,emb_size,bias=False)
        self.cache_adj=cache_adj
        self.normalized_adj=None

        
    
    def forward(self, adj, feature):
        if self.cache_adj and self.normalized_adj is not None:
            normalized_adj=self.normalized_adj
        else:
            row_sum=torch.sparse.sum(adj,dim=1)
            row_sum=1.0 / (row_sum.to_dense()+1e-10)
            row_sum=row_sum.masked_fill(row_sum==1e10,0.0)
           
            n = row_sum.shape[0]
            # Create sparse tensor with same shape and nnz equal to number of elements  
            row_sum = torch.sparse_coo_tensor(indices=torch.stack((torch.arange(n).to(row_sum.device),torch.arange(n).to(row_sum.device))), 
                                                    values=row_sum, 
                                                    size=(n,n))
            
            normalized_adj = torch.sparse.mm(row_sum, adj)
            self.normalized_adj=normalized_adj
        if self.with_linear:
            return self.linear( torch.sparse.mm(normalized_adj, feature))
        else:
            return torch.sparse.mm(normalized_adj, feature)
        

    def __init__(self,with_linear=False,emb_size=None,cache_adj=True):
        super(HighPassingMeanAggregator, self).__init__()
        self.with_linear=with_linear
        self.betas=nn.Parameter(torch.tensor([[1.0]*64]))
        print(self.betas.shape) 
        if with_linear:
            self.linear=nn.Linear(emb_size,emb_size,bias=False)
        self.cache_adj=cache_adj
        self.normalized_adj =None
        
    
    def forward(self, adj, feature):
        if self.cache_adj and self.normalized_adj is not None:
            normalized_adj=self.normalized_adj
        else:
            row_sum=torch.sparse.sum(adj,dim=1)
            row_sum=1.0 / (row_sum.to_dense()+1e-10)
           
            row_sum=row_sum.masked_fill(row_sum==1e10,0.0)
           
            n = row_sum.shape[0]
            # Create sparse tensor with same shape and nnz equal to number of elements  
            row_sum = torch.sparse_coo_tensor(indices=torch.stack((torch.arange(n).to(row_sum.device),torch.arange(n).to(row_sum.device))), 
                                                    values=row_sum, 
                                                    size=(n,n))
            I = torch.sparse_coo_tensor(indices=torch.stack((torch.arange(n).to(row_sum.device),torch.arange(n).to(row_sum.device))), 
                                                    values=torch.tensor([1]*n).to(row_sum.device), 
                                                    size=(n,n))
            
            normalized_adj = torch.sparse.mm(row_sum, adj)
          
            self.normalized_adj=normalized_adj
            print(normalized_adj)
        if self.with_linear:
            return self.linear( torch.sparse.mm(normalized_adj, feature))
        else:
            return feature- torch.sparse.mm(normalized_adj, feature)
        
class GCNAggregator(nn.Module):
    def __init__(self,with_linear=False,emb_size=None,cache_adj=True):
        super(GCNAggregator, self).__init__()
        self.with_linear=with_linear
        if with_linear:
            self.linear=nn.Linear(emb_size,emb_size,bias=False)

        
    
    def forward(self, adj, feature):
        
        if self.with_linear:
            return self.linear( torch.sparse.mm(adj, feature))
        else:
            return torch.sparse.mm(adj, feature)



        



  
