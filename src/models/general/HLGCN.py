

import torch
import numpy as np
import logging
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from utils.graphlayer import MeanAggregator,GCNAggregator
from tqdm import tqdm
from models.BaseModel import GeneralModel


class HLGCN(GeneralModel):
    extra_log_args = ['emb_size', 'n_layers','w_linear','encoder_type']
    reader='HLGCNReader'
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of LightGCN layers.')
        parser.add_argument('--encoder_type', type=str, default='Adj',
                            help='Adj,Mean,Atten')
        parser.add_argument('--w_linear',type=int,default=1)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.encoder_type=args.encoder_type
        self.w_linear=args.w_linear
        self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set,self.encoder_type,selfloop_flag=True)
      
        self.norm_high_pass_adj = self.build_high_pass_adjmat(corpus.n_users, corpus.n_items, corpus.train_dis_clicked_set,self.encoder_type,selfloop_flag=False)
        self.norm_pass_adj_neg = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_dis_clicked_set,self.encoder_type,selfloop_flag=False)
        
        logging.info(str(len(corpus.train_clicked_set))+' dis len: '+str(len(corpus.train_dis_clicked_set)))

        super().__init__(args, corpus)

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat,encoder_type,selfloop_flag=False):
        R = sp.dok_matrix((user_count+item_count, user_count+item_count), dtype=np.float32)
        for user in tqdm(train_mat):
            for item in train_mat[user]:
                R[user, item+user_count] = 1
                R[ item+user_count,user] = 1

        # RT = sp.dok_matrix((item_count,user_count), dtype=np.float32)
        # for user in train_mat:
        #     for item in train_mat[user]:
        #         RT[item, user] = 1
        # RT = RT.tolil()
        # adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        # adj_mat = adj_mat.tolil()

        # adj_mat[:user_count, user_count:] = R
        # adj_mat[user_count:, :user_count] = RT
        adj_mat = R

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1)) + 1e-10

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        if selfloop_flag:
            if encoder_type=='Adj':
                norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            else:
                norm_adj_mat=adj_mat + sp.eye(adj_mat.shape[0])
        else:
            if encoder_type=='Adj':
                norm_adj_mat = normalized_adj_single(adj_mat)
            else:
                norm_adj_mat=adj_mat

        return norm_adj_mat.tocsr()

    @staticmethod
    def build_high_pass_adjmat(user_count, item_count, train_mat,encoder_type,selfloop_flag=False):
        R = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, user_count+item] = 1
                R[user_count+item,user] = 1
        adj_mat=R

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1)) + 1e-10
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            # d_inv_sqrt[d_inv_sqrt>=1e5]=0.
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            d_mat=sp.diags(rowsum.flatten())
         

            bi_lap = d_mat_inv_sqrt.dot((d_mat-adj)).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()
        

        if selfloop_flag:
            if encoder_type=='Adj':
                norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            else:
                norm_adj_mat=adj_mat + sp.eye(adj_mat.shape[0])
        else:
           
            if encoder_type=='Adj':
                norm_adj_mat = normalized_adj_single(adj_mat)
            else:
                norm_adj_mat=adj_mat


        return norm_adj_mat.tocsr()

    def _define_params(self):
        self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj,self.norm_high_pass_adj,self.norm_pass_adj_neg,self.encoder_type,self.w_linear, self.n_layers,self.device)
        

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
        out_dict = {'prediction': prediction}
        return out_dict
    def save_encoded_embedding(self):
        u_emb,i_emb=self.encoder(torch.arange(self.user_num,device=self.device), torch.arange(self.item_num,device=self.device))
      
        u_emb_path=self.model_path.replace('.pt','u_emb.emb')
        i_emb_path=self.model_path.replace('.pt','i_emb.emb')
        torch.save(u_emb.detach().cpu(), u_emb_path)
        torch.save(i_emb.detach().cpu(), i_emb_path)
        print(u_emb_path,i_emb_path)

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

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings
    






