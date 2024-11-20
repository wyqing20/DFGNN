import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel
from models.general.HLGCN import HLGCN

class HLGCNReg(HLGCN):
    extra_log_args = ['emb_size', 'n_layers','w_linear','encoder_type']
    runner='RegRunner'
    reader='SignedReader'


    
    
    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        BPR ranking loss with optimization on multiple negative samples (a little different now)
        "Recurrent neural networks with top-k gains for session-based recommendations"
        :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        predictions = out_dict['prediction']
        labels=out_dict['label']
        pos_ratio = labels.sum() /  labels.size()[0]
        # weight = torch.where(labels > 0.5, 1-pos_ratio,pos_ratio)
        loss=F.binary_cross_entropy(predictions, labels.float())
       
        # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        # loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # ↑ For numerical stability, we use 'softplus(-x)' instead of '-log_sigmoid(x)'
        return loss
    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        
        u_embed, i_embed = self.encoder(user, items)

        prediction = (u_embed * i_embed).sum(dim=-1).sigmoid()
        out_dict = {'prediction': prediction,'label':feed_dict['label']}
        
        return out_dict


    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            feed_dict = {
                'user_id': user_id,
                'item_id': target_item,
                'label':self.data['label'][index]
            }
            return feed_dict
        
        def actions_before_epoch(self):
            pass
        
        