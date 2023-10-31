# import needed library
import os
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils import get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC

from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50, WR50_WR50
import warnings

warnings.filterwarnings('ignore')


def main_worker(gpu, args):
    '''
    '''

    args.gpu = gpu
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    # cudnn.benchmark = True

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "INFO"
    tb_log = None

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    # Construct Dataset & DataLoader
    train_dset = AD_Dataset(name=args.dataset, train=True, data_dir=args.data_dir)
    train_dset = train_dset.get_dset()
    print('TrainSet Image Number:', len(train_dset))
    eval_dset = AD_Dataset(name=args.dataset, train=False, data_dir=args.data_dir)
    eval_dset = eval_dset.get_dset()
    print('EvalSet Image Number:', len(eval_dset))

    loader_dict = {}
    dset_dict = {'train': train_dset, 'eval': eval_dset}

    generator_lb = torch.Generator()
    generator_lb.manual_seed(args.seed)
    loader_dict['train'] = get_data_loader(dset_dict['train'],
                                           args.batch_size,
                                           data_sampler=args.train_sampler,
                                           num_iters=args.num_train_iter,
                                           num_workers=args.num_workers,
                                           distributed=False,
                                           generator=generator_lb)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size,
                                          num_workers=args.num_workers,
                                          drop_last=False)

    model = R50_R50(img_size=args.img_size,
                    train_encoder=True,
                    stop_grad=True,
                    reshape=True,
                    bn_pretrain=False,
                    )

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    runner = EDC(model=model,
                 num_eval_iter=args.num_eval_iter,
                 tb_log=tb_log,
                 logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(runner.model)}')

    # SET Optimizer & LR Scheduler
    optimizer = get_optimizer_v2(runner.model, args.optim, args.lr, args.momentum, lr_encoder=args.lr_encoder,
                                 weight_decay=args.weight_decay)
    scheduler = get_multistep_schedule_with_warmup(optimizer, milestones=[1e10], gamma=0.2,
                                                   num_warmup_steps=0)
    runner.set_optimizer(optimizer, scheduler)

    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        runner.model = runner.model.cuda(args.gpu)

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    ## set DataLoader
    runner.set_data_loader(loader_dict)
    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        runner.load_model(args.load_path)

    # START TRAINING
    runner.tb_log = TBLog(save_path, 'tb', use_tensorboard=args.use_tensorboard)
    runner.train(args)
    logging.warning(f"training is FINISHED")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str,
                        default='edcad_aptos_256_r50_r50_m4_bn99_adamw5e4wd1e4_1e5_b32_i1k_cl1_0',
                        )
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''  
    Training Configuration
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=1000,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=250,
                        help='evaluation frequency')
    parser.add_argument('-bsz', '--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=0., help='ema momentum for eval_model')

    ''' 
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_encoder', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=1)
    ''' 
    Data Configurations
    '''
    parser.add_argument('--data_dir', type=str, default="/data/disk2T1/guoj/APTOS")
    parser.add_argument('-ds', '--dataset', type=str, default='fundus')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', default='1', type=str,
                        help='GPU id to use.')

    # config file
    parser.add_argument('--c', type=str, default='')
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil

        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    main_worker(int(args.gpu), args)
