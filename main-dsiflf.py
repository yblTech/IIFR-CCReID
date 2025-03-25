import os
import time
import datetime
import argparse
import os.path as osp
import shutil
import torch
import torch.nn as nn
from torch import distributed as dist
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from configs.default_img import get_img_config
from data import build_dataloader
from models import build_model
from tools.utils import save_checkpoint, set_seed, get_logger
from test import  test2, test_prcc2
from train import  train_adv2
from losses import build_losses


def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_img_config(args)

    return config,args
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def main(config,dpath):
    # Build dataloader
    if config.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(config, only_dsiflf=True)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config, only_dsiflf=True)
    pid2clothes = torch.from_numpy(dataset.pid2clothes)
    print(dpath)
    # Build model
    model, classifier,_ = build_model(config, dataset.num_train_pids, dataset.num_train_clothes, OnlyDSIFLF=True)             
    # Build loss    
    criterion_cla, criterion_prototype, criterion_cal, criterion_cicl, criterion_tri = build_losses(config, dataset.num_train_clothes, dataset.num_train_pids)
    # Build optimizer
    
    parameters = list(model.parameters()) + list(classifier.parameters()) 
    lr_min = 3.5e-6
    lr_max = 3.5e-4
    step_size_up = 10
    step_size_down = [40, 80]
    gamma = 0.1

    def lr_lambda(current_step):
        if current_step < step_size_up:
            return (lr_max - lr_min) / step_size_up * current_step + lr_min
        else:
            factor = 1
            for s in step_size_down:
                if current_step > s:
                    factor *= gamma
            return lr_max * factor
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        # optimizer = optim.Adam(parameters, lr=1)
        # optimizer_cc = optim.Adam(parameters2, lr=1)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    
    
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)
    # scheduler = LambdaLR(optimizer, lr_lambda)
    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'])
        # start_epoch = checkpoint['epoch']
        
    local_rank = dist.get_rank()
    model = model.cuda(local_rank)  
    pid2clothes = pid2clothes.cuda(local_rank)
    classifier = classifier.cuda(local_rank)
    torch.cuda.set_device(local_rank)


    if config.TRAIN.AMP:
        [model, classifier], optimizer = amp.initialize([model, classifier], optimizer, opt_level="O1")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc2(config,model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test2(config, model, queryloader, galleryloader, dataset)
                # tsne(config, model, queryloader, galleryloader, dataset)
                # cam(model, discriminator)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = 45
    best_epoch = 0
    logger.info("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        start_train_time = time.time()

        train_adv2(config, epoch, model, classifier, criterion_cla, criterion_prototype, criterion_cicl, 
                   criterion_tri, optimizer,trainloader, pid2clothes)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 or epoch == 4 or epoch == 9 or epoch == 14 or epoch == 0:
            logger.info("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc':
                rank1 = test_prcc2(config,model, queryloader_same, queryloader_diff, galleryloader, dataset)
                
            else:
                rank1 = test2(config, model, queryloader, galleryloader, dataset)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            model_state_dict = model.module.state_dict()
            if local_rank == 0 and is_best:
                save_checkpoint({
                    'model_state_dict': model_state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(dpath, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()

if __name__ == '__main__':
    config,args = parse_option()
    current_time = datetime.datetime.now()
    current_date_str = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    dpath = osp.join(config.OUTPUT, current_date_str)
    os.makedirs(dpath, exist_ok=True)
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    print(dpath)
    set_seed(config.SEED + local_rank)
    # get logger
    if not config.EVAL_MODE:
        output_file = osp.join(dpath, 'log_train.log')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.log')
    yp=osp.join(osp.dirname(os.path.abspath(__file__)), args.cfg)
    dp="configs/default_img.py"
    source_file=["main-dsiflf.py","test2.py","train.py",
    yp, dp]
    source_dir=["tools","models","losses",
    "data"]
    for i in range(len(source_file)):
        shutil.copy2(source_file[i], dpath)
    for i in range(len(source_dir)):
        dpathdir=osp.join(dpath,source_dir[i].split('/')[-1])
        shutil.copytree(source_dir[i], dpathdir, dirs_exist_ok=True)
    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    main(config,dpath)