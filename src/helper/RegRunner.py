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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner
from sklearn.metrics import roc_auc_score
from imblearn.metrics import geometric_mean_score



class RegRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=4000,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=10,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=512,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=512,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='10,5,20,50',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='AUC,F1_Macro,F1,POS_R,F1_Micro',
                            help='metrics: NDCG, HR')
        parser.add_argument('--l_weight', type=float, default=0.1,
                            help='Number of epochs.')
        parser.add_argument('--eval_type', type=str, default='batch',
                            help='batch eval or not batch')
        return parser
    def __init__(self, args):
        super().__init__(args)
        self.main_metric='AUC'
    
    def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions,labels = self.predict(data)
        predictions,labels=predictions.numpy(),labels.numpy()

        # 预测标签列表和真实标签列表
     
        # 计算AUC
        
        auc = roc_auc_score(labels, predictions)
        if auc<0.5:
            auc=1-auc
           
        predictions=np.where(predictions>=0.5,1,0)
        # 计算ACC
        # auc = roc_auc_score(labels, predictions)
 
        
        # 计算F1分数
        f1 = f1_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        micro_f1 = f1_score(labels, predictions, average='micro')
        evaluations=dict()
        evaluations['AUC']=auc
        
        evaluations['F1_Macro']=macro_f1
        evaluations['F1_Micro']=micro_f1
        evaluations['AUC']=auc
        evaluations['F1']=f1
        evaluations['pos_r']=predictions.sum()/predictions.shape[0]

        return evaluations

    def predict(self, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
        labels=list()
        predictions = list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        with torch.no_grad():
            for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
                prediction = data.model.inference(utils.batch_to_gpu(batch, data.model.device))
                label=prediction['label'].cpu().data
                prediction=prediction['prediction']
                predictions.append(prediction.cpu().data)
                labels.append(label)
            
            
        predictions=torch.cat(predictions,dim=0)
        labels=torch.cat(labels)
        return predictions,labels