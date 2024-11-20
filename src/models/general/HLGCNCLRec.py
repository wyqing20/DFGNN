import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
import logging
from models.BaseModel import GeneralModel
from models.general.HLGCNCL import HLGCNCL

from utils.graphlayer import MeanAggregator,GCNAggregator
class HLGCNCLRec(HLGCNCL):
    
    runner='RecRunner'
    reader='SignedReader'
 

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        print(model_path)
        model_path=model_path.replace("model","model_br")
        self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)

   
    def _define_params(self):
        self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj,self.norm_high_pass_adj,self.norm_pass_adj_neg,self.encoder_type,self.w_linear, self.n_layers,self.device)
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
        u_embed, i_embed = self.encoder(user, items)
        
        u_embed, i_embed=u_embed[user[pos_edges]],i_embed[rec_items]
        # u_embed,i_embed=[self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']]
        
        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1).sigmoid()
        edges=torch.stack([user,items],dim=0).T
        out_dict = {'prediction': prediction,'label':feed_dict['label'],'edges':edges}
      
        return out_dict
    def inference(self, data):
        
        users=data
        u_embed, i_embed = self.encoder(None, None)
        # u_embed,i_embed=[self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']]
        u_embed=u_embed[users]  
        prediction_matrix=u_embed.mm(i_embed.T).sigmoid()
        return {'prediction': prediction_matrix}
    def inference_all(self):
        u_embed, i_embed = self.encoder(None, None)
        prediction_matrix=u_embed.mm(i_embed.T)
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

class LGCNEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj,norm_high_pass_adj, norm_pass_adj_neg,encoder_type,w_linear, n_layers=3,device=None):
        super(LGCNEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.encoder_type=encoder_type
        self.emb_size = emb_size
        self.w_linear=w_linear
        self.layers = [emb_size] * n_layers
        self.norm_adj = norm_adj
        self.n_layers=n_layers
        self.norm_high_pass_adj=norm_high_pass_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(device)
        self.sparse_norm_high_pass_adj=self._convert_sp_mat_to_sp_tensor(self.norm_high_pass_adj).to(device)
        self.eye_adj=self._convert_sp_mat_to_sp_tensor(sp.eye(self.norm_adj.shape[0])).to(device)
        self.norm_neg_pass_adj=self._convert_sp_mat_to_sp_tensor(norm_pass_adj_neg).to(device)
        self.gcn_layers= nn.ModuleList([nn.Sequential(nn.Linear(self.emb_size*2,self.emb_size*1,bias=True)) for _ in range(n_layers)])
        self.layer_norm_layers1=nn.ModuleList([nn.Sequential(nn.LayerNorm(self.emb_size)) for _ in range(n_layers-1)])
        self.layer_norm_layers2=nn.ModuleList([nn.Sequential(nn.LayerNorm(self.emb_size)) for _ in range(n_layers-1)])
        if self.encoder_type=='Adj':
            self.low_aggs=nn.ModuleList([GCNAggregator() for _ in range(n_layers)])
            self.high_aggs=nn.ModuleList([GCNAggregator() for _ in range(n_layers)])
        elif self.encoder_type=='Mean':
            self.low_aggs=nn.ModuleList([MeanAggregator() for _ in range(n_layers)])
            self.high_aggs=nn.ModuleList([MeanAggregator() for _ in range(n_layers)])
        elif self.encoder_type=='Atten':
            self.low_aggs=nn.ModuleList([AttentionAggregator(self.emb_size,self.emb_size) for _ in range(n_layers)])
            self.high_aggs=nn.ModuleList([AttentionAggregator(self.emb_size,self.emb_size) for _ in range(n_layers)])



    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
       
        self.betas=nn.ParameterList([nn.Parameter(torch.tensor([[0.1]])) for _ in range(self.n_layers)] )
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, users, items):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        pos_ego_embeddings=ego_embeddings
        dis_ego_embeddings=ego_embeddings
        for k in range(len(self.layers)):
            pos_ego_embeddings = self.low_aggs[k](self.sparse_norm_adj, pos_ego_embeddings)
            dis_ego_embeddings=self.high_aggs[k](self.sparse_norm_high_pass_adj,dis_ego_embeddings)
           
            ego_embeddings=torch.cat([pos_ego_embeddings,dis_ego_embeddings],dim=1)
           
         
            if self.w_linear!=0:
            # ego_embeddings=pos_ego_embeddings
                ego_embeddings=self.gcn_layers[k](ego_embeddings)
          
           
            all_embeddings += [ego_embeddings]
            pos_ego_embeddings=ego_embeddings
            dis_ego_embeddings=ego_embeddings
          
         
        # all_embeddings = torch.stack(all_embeddings, dim=1)
        # all_embeddings = torch.mean(all_embeddings, dim=1)
        # print(ego_embeddings)
        all_embeddings=ego_embeddings
        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]


        return user_all_embeddings, item_all_embeddings
 


    









        
         
