# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch

from helpers import *
from models.general import *
from utils import utils


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='1',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=2024,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=1,
                        help='Whether to regenerate intermediate files.')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
               'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('GPU available: {}'.format(torch.cuda.is_available()))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    if  True and args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
        logging.info('Loaded corpus from {}'.format(corpus_path))
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    # Define model
    model = model_name(args, corpus)
    logging.info(model)
    if 'SBGCL' in args.model_name2:
        model.apply(model.init_weights)
        pass
    else:
        
        model.apply(model.init_weights)
    model.actions_before_train()
    model.to(model.device)

    # Run model
    data_dict = dict()
    phases= ['train', 'dev', 'test']
   
    phases_real=['train', 'dev', 'test']
    
    if hasattr(model,'stage') and model.stage!=1 and model.stage!=6 and model.stage!=8 and model.stage!=10 and model.stage<20 and model.stage<=30:
        
     
        phases_real=['cold_train2', 'cold_dev2', 'cold_test2']

        
        
    
    
    for phase,phase_real in zip(phases,phases_real):
       
        data_dict[phase] = model_name.Dataset(model, corpus, phase_real)
    print(data_dict['dev'])
    runner = runner_name(args)
    # logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))
    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(data_dict)
    logging.info(os.linesep + 'Test After Training: ' + runner.print_res(data_dict['test']))
    try:
        pass
     
    except Exception as e:
        print(e)
    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)



if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='Bert4rec', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    runner_name = eval('{0}.{0}'.format(model_name.runner))
  
    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()
    args.model_name2=init_args.model_name
    random_seeds=[2024]
 
    for random_seed in random_seeds:
  
        log_args = [init_args.model_name, args.dataset, str(args.random_seed)]

        model_path_args=[init_args.model_name, args.dataset, str(args.random_seed)]
        not_in_logs=['weight','temp']
        print(model_name.extra_log_args)
       

        for arg in ['lr', 'l2'] + model_name.extra_log_args:
            if arg not in not_in_logs:
                log_args.append(arg + '=' + str(eval('args.' + arg)))
            else:
                weight=eval('args.' + 'weight')
                
                if weight==0.0:
                    log_args.append(arg + '=' + str(eval('args.' + arg)))

            model_path_args.append(arg + '=' + str(eval('args.' + arg)))
        log_file_name = '__'.join(log_args).replace(' ', '__')
        model_file_name='__'.join(model_path_args).replace(' ', '__')
        if args.log_file == '':
            args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
        if args.model_path == '':
            args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, model_file_name)
        print(args.model_path)
        utils.check_dir(args.log_file)
        logging.basicConfig(filename=args.log_file, level=args.verbose)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(init_args)

        main()
