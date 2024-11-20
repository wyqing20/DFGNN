import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel
from models.general.LPGCNCL import LPGCNCL

class LPGCNCLRec(LPGCNCL):
    extra_log_args = ['emb_size', 'n_layers','w_linear','encoder_type' ,'encoder', 'weight', 'temp']
    runner='RecRunner'
    reader='SignedReader'
    

    
    
    def rec_loss(self, out_dict: dict) -> torch.Tensor:
        predictions = out_dict['prediction']
        lables=torch.ones_like(predictions).to(predictions.device)
        lables[:,1:]=0
        predictions,lables=predictions.flatten(),lables.flatten()
        loss=F.binary_cross_entropy(predictions, lables.float())
        return loss
    def loss(self, out_dict: dict) -> torch.Tensor:
        return self.rec_loss(out_dict)+self.weight*self.CL_loss(out_dict['edges'],out_dict['label'])
        
       
        
    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        labels=feed_dict['label']
        pos_edges=labels==1
        rec_pos_items,rec_neg_items=items[pos_edges].unsqueeze(-1),feed_dict['neg_items'][pos_edges]
        rec_items=torch.cat([rec_pos_items,rec_neg_items],dim=1)
        u_embed, i_embed = self.encoder(torch.arange(self.user_num,device=user.device), torch.arange(self.item_num,device=user.device))
        
        u_embed, i_embed=u_embed[user[pos_edges]],i_embed[rec_items]
        # u_embed,i_embed=[self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']]
        
        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1).sigmoid()
        edges=torch.stack([user,items],dim=0).T
        out_dict = {'prediction': prediction,'label':feed_dict['label'],'edges':edges}
      
        return out_dict
    def inference(self, data):
        
        users=data
      
        u_embed, i_embed = self.encoder(torch.arange(self.user_num,device=users.device), torch.arange(self.item_num,device=users.device))
        # u_embed,i_embed=[self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']]
        u_embed=u_embed[users]  
        prediction_matrix=u_embed.mm(i_embed.T).sigmoid()
        return {'prediction': prediction_matrix}


    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase: str):
            super().__init__(model, corpus, phase)
            self.buffer=0
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            
            feed_dict = {
                'user_id': user_id,
                'item_id': target_item,
                'label': self.data['label'][index],
                
            }
            if 'train' in self.phase:
                feed_dict['neg_items'] = self.data['neg_items'][index]
            return feed_dict
        
        def actions_before_epoch(self):
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
          
            for i, u in enumerate(self.data['user_id']):
                
                clicked_set = self.corpus.train_clicked_set.get(u,set())  # neg items are possible to appear in dev/test set
                for j in range(self.model.num_neg):
                    while neg_items[i][j] in clicked_set:
                        neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
            self.data['neg_items'] = neg_items
        
        