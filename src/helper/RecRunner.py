# -*- coding: UTF-8 -*-

import os
import gc
from numpy.testing._private.utils import print_assert_equal
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn


from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner

from imblearn.metrics import geometric_mean_score
from evaluator.metrics import  Hit,NDCG,MRR
import recsys_metrics



class RecRunner(BaseRunner):
    

    @staticmethod
    def get_ranks(predictions,lable):
    
 
   
        gt_ranks=[]
        all_ranks=[]
        rows=predictions.shape[0]
        
        
        for i in range((rows-1)//1000+1):
            start=i*1000
           
            end=min(i*1000+1000,rows)
            batch_rank=(-predictions[start:end,:]).argsort(dim=1).argsort(dim=1)+1
            batch_lable=lable[start:end,:]
           
            indices= torch.nonzero(batch_lable == 1).squeeze()
            
            grouped_indices = {}
            indices=indices.cpu().numpy()
            for idx in indices:
                
            
                row_idx,col_idx=int(idx[0]),int(idx[1])
            
                if row_idx not in grouped_indices:
                    grouped_indices[row_idx] = []
                grouped_indices[row_idx].append(col_idx)
            batch_gourped_ranks=[]
            for row in range(len(batch_rank)):
                batch_gourped_ranks.append(batch_rank[row][grouped_indices[row]].cpu().numpy())
            
            
            gt_ranks.append(batch_gourped_ranks)

        gt_ranks=[j for i in gt_ranks for j in i]

        return    gt_ranks
    @staticmethod
    def evaluate_method(predictions: Dict[str,torch.Tensor], topk: list, metrics: list) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()
        predictions,labels=predictions['predictions'],predictions['label']
        # print(predictions[0])
        batch_size,score_len=predictions.shape[0],predictions.shape[1]
        # print(predictions.shape,labels.shape)
        for metric in metrics:
            max_topk=max(topk)
            top_predictions,topk_index=predictions.topk(max_topk)
            pos_index,pos_len=labels.take_along_dim(topk_index, dim=-1),labels.sum(dim=-1).long()
            if metric == 'HIT':
                HR=Hit(config={'topk':topk,'metric_decimal_place':4})
                res=HR.calculate_metric((pos_index,pos_len))
                for key in res:
                    evaluations[key.upper()] = res[key]
            elif metric == 'NDCG':
                ndcg=NDCG(config={'topk':topk,'metric_decimal_place':4})
                res=ndcg.calculate_metric((pos_index,pos_len))
                key = '{}@{}'.format(metric, topk[0])
                for key in res:
                    evaluations[key.upper()] = res[key]
            else:
                pass
                # raise ValueError('Undefined evaluation metric: {}.'.format(metric))
       
        

        
        return evaluations
    

    def evaluate_batch(self,predictions,users,ui_dict,mask_set,topk,device):
         
        rows, cols = list(), list()
        users=users.numpy()
        for i, u in enumerate(users):
            clicked_items = [x for x in mask_set.get(u,[])]
            # clicked_items = [data.data['item_id'][i]]
            idx = list(np.ones_like(clicked_items) * i)
            rows.extend(idx)
            cols.extend(clicked_items)
        predictions[rows, cols] = -np.inf
        
        inf_nums=(predictions==-np.inf).sum().item()
        rows,cols=list(),list()
        
        for i,user in enumerate(users):
            item,label=ui_dict[user]
            mask=label==1
           
            item=item[mask]
            
            rows.append(torch.tensor([i]*len(item)))
            cols.append(torch.from_numpy(item))
        rows,cols=torch.cat(rows,dim=0),torch.cat(cols,dim=0)
        n_users,n_items=predictions.shape
        labels=torch.zeros((n_users,n_items),dtype=torch.long)
        labels[rows,cols]=1

        labels,predictions=labels[:,1:],predictions[:,1:]
        predictions,labels=predictions.to(device),labels.to(device)
        grouped_ranks=RecRunner.get_ranks(predictions,lable=labels)
        
        max_topk=max(topk)
        top_predictions,topk_index=predictions.topk(max_topk)
        pos_index,pos_len=labels.take_along_dim(topk_index, dim=-1),labels.sum(dim=-1).long()
        return grouped_ranks,pos_index,pos_len
        
    def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list):
        
        if self.eval_type=='batch':
            grouped_ranks,pos_indexs,pos_lens = self.prediction_batch(data,topks)
            pos_indexs,pos_lens=pos_indexs.cpu(),pos_lens.cpu()
          
            return self.summary_batched_metric(grouped_ranks,pos_indexs=pos_indexs,pos_lens=pos_lens,num_items=data.corpus.n_items-1,metrics=metrics,topk=topks)
        else:
            return self.evaluate_all(data,topks,metrics)

    def evaluate_all(self, data: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(data)
        all_users=data.data['user_id'][data.data['label']==1]
        all_users=np.sort(np.unique(all_users))
        print(data.model.device)
        print(predictions.mean(),predictions.std())
   
        if data.model.test_all or True:
            rows, cols = list(), list()
            for i, u in enumerate(all_users):
                clicked_items = [x for x in data.corpus.train_clicked_set.get(u,[])]
                # clicked_items = [data.data['item_id'][i]]
                idx = list(np.ones_like(clicked_items) * i)
                rows.extend(idx)
                cols.extend(clicked_items)
                # clicked_items = [x for x in data.corpus.train_dis_clicked_set.get(u,[])]
                # idx = list(np.ones_like(clicked_items) * i)
                # rows.extend(idx)
                # cols.extend(clicked_items)
            predictions[rows, cols] = -np.inf
        print((predictions==-np.inf).sum())
        rows,cols,label=data.data['user_id'],data.data['item_id'],data.data['label']
        mask=label==1
        rows,cols=rows[mask],cols[mask]
        # print(rows.shape)
        label=torch.zeros((data.corpus.n_users,data.corpus.n_items),dtype=torch.long)
        label[rows,cols]=1
        
        label=label[all_users]
        print(predictions.shape,label.shape)
        label,predictions=label[:,1:],predictions[:,1:]
        for i in [1,2,3,4,5,10,20,50]:
            aa=recsys_metrics.rank_report(predictions,label,k=i)
            print(i,aa)
        # print(aa)
        predictions={'predictions':predictions,'label':label}
        return self.evaluate_method(predictions, topks, metrics)

    def predict(self, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
        predict_users=data.data['user_id'][data.data['label']==1]
        predict_users=np.sort(np.unique(predict_users))
        predict_users=torch.from_numpy(predict_users)
        dl = DataLoader(predict_users, batch_size=self.eval_batch_size, shuffle=False)
        predictions=list()
        with torch.no_grad():
            
            for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
                prediction = data.model.inference(batch.to(data.model.device))
                prediction=prediction['prediction']
                
                predictions.append(prediction.cpu().data)
            
        
        predictions=torch.cat(predictions,dim=0)
        print(prediction.shape)
        # print(len(predictions),len(predict_users))
        return predictions
    
    def summary_batched_metric(self,gt_ranks,pos_indexs,pos_lens,num_items,metrics,topk):

        evaluations = dict()
      
        if 'MRR' in metrics:
            max_rank=[min(i) for i in gt_ranks]
            mrr=(1/np.array(max_rank)).mean()
                
            evaluations['MRR']=mrr
        for metric in metrics:
            max_topk=max(topk)
            if metric == 'HIT':
                HR=Hit(config={'topk':topk,'metric_decimal_place':4})
                res=HR.calculate_metric((pos_indexs,pos_lens))
                for key in res:
                    evaluations[key.upper()] = res[key]
            elif metric == 'NDCG':
                ndcg=NDCG(config={'topk':topk,'metric_decimal_place':4})
                res=ndcg.calculate_metric((pos_indexs,pos_lens))
                key = '{}@{}'.format(metric, topk[0])
                for key in res:
                    evaluations[key.upper()] = res[key]
            else:
                pass
        return evaluations
    def prediction_batch(self,data:BaseModel.Dataset,topks):
        pass
        data.model.eval()
        u_i_dict={}
        for u,i ,label in zip(data.data['user_id'],data.data['item_id'],data.data['label']):
            if u not in u_i_dict:
                u_i_dict[u]=[]
            u_i_dict[u].append((i,label))
        for k,v in u_i_dict.items():
            u_i_dict[k]=np.array([x[0] for x in v]),np.array([x[1] for x in v])
        predict_users=data.data['user_id'][data.data['label']==1]
        predict_users=np.sort(np.unique(predict_users))
        predict_users=torch.from_numpy(predict_users)
        dl = DataLoader(predict_users, batch_size=self.eval_batch_size, shuffle=False)
        predictions=list()
        grouped_ranks=list()
        pos_indexs=list()
        pos_lens=list()
        with torch.no_grad():
            
            for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
                prediction = data.model.inference(batch.to(data.model.device))
                prediction=prediction['prediction']
                
                grouped_rank,pos_index,pos_len=self.evaluate_batch(prediction,users=batch,ui_dict=u_i_dict,mask_set=data.corpus.train_clicked_set,topk=topks,device=data.model.device)
                grouped_ranks.append(grouped_rank)
                pos_indexs.append(pos_index)
                pos_lens.append(pos_len)
                # predictions.append(prediction.cpu().data)
            
        pos_indexs=torch.cat(pos_indexs,dim=0)
        pos_lens=torch.cat(pos_lens,dim=0)
        grouped_ranks=[j for i in grouped_ranks for j in i]
        return grouped_ranks,pos_indexs,pos_lens
    def train(self, data_dict: Dict[str, BaseModel.Dataset]) -> NoReturn:
        model = data_dict['train'].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True) 
        # print([model.encoder.betas[i].sigmoid() for i in range(len(model.encoder.betas))])
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                if self.batch_size==-1:
                    self.batch_size=len(data_dict['train'])
                    self.eval_batch_size=len(data_dict['train'])
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)
                # print([model.encoder.betas[i].sigmoid() for i in range(len(model.encoder.betas))])
                # Record dev results
                eval_metric=self.metrics
                dev_result = self.evaluate(data_dict['dev'], self.topk[:10], eval_metric)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    dev=({})'.format(
                    epoch + 1, loss, training_time, utils.format_metric(dev_result))
               
                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
                    test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == 4):
                    model.save_model()
                    logging_str += ' *'
                
                logging.info(logging_str)
               
                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
            best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()