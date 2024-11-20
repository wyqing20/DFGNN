


import torch
import numpy as np

import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from utils.graphlayer import MeanAggregator,GCNAggregator

from models.BaseModel import GeneralModel
from models.general.LPGCN import LPGCN

class LPGCNCL(LPGCN):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--temp', type=float, default=1,
                            help='temperature of CL')
        parser.add_argument('--weight', type=float, default=0.0,
                            help='weight of CL')
        return LPGCN.parse_model_args(parser)
       
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.temp=args.temp
        self.weight=args.weight
    def _define_params(self):
        super()._define_params()
        self.contrst1=nn.Linear(self.emb_size,self.emb_size)
        self.contrst2=nn.Linear(self.emb_size,self.emb_size)

    def loss(self, out_dict: dict) -> torch.Tensor:
        return super().loss(out_dict)+self.weight*self.CL_loss(out_dict['edges'],out_dict['label'])
    def CL_loss(self,edges,labels):
        if labels.shape[0]==1:
            return 0.0
        a_embedings=self.encoder.embedding_dict['user_emb'][edges[:,0]]
        b_embedings=self.encoder.embedding_dict['item_emb'][edges[:,1]]
      
        scores=F.cosine_similarity(a_embedings.unsqueeze(1), b_embedings.unsqueeze(0), dim=2)/self.temp
       
        labels=labels*2-1
       
        diagonal_scores=torch.diag(scores)
       
        scores=torch.exp(scores)
       
        diagonal_scores=-labels*diagonal_scores
        
        scores=scores.sum(-1).log()
        
        loss=diagonal_scores.mean()+scores.mean()
        
        
        

    

        return loss
    
    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
        edges=torch.stack([user,items],dim=0).T
        out_dict = {'prediction': prediction,'label':feed_dict['label'],'edges':edges}
        return out_dict
    





        
        
        