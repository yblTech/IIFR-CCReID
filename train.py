import time
import datetime
import logging
import torch
import torch.nn as nn
import random
import copy
# from apex import amp
from tools.utils import AverageMeter
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init
from models.Prototype import Prototype
from torchvision import transforms as T

def train_adv(config, epoch, model, genertor, Discriminator, classifier, num_train_pids, criterion_cla, criterion_cal, 
        criterion_prototype, criterion_cicl, criterion_tri, optimizer, optimizer_cc, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_DiscLoss = AverageMeter()
    batch_TriLoss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_GenLoss = AverageMeter()
    batch_CiclLoss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    camadv_loss = nn.BCEWithLogitsLoss()
    CycleLoss = nn.MSELoss()
    model.train()
    classifier.train()
    Discriminator.train()
    end = time.time()
    softmax = nn.Softmax(dim=1)
    p1 = Prototype(num_pids=num_train_pids, feat_dim=config.MODEL.FEATURE_DIM, margin=0.3, 
                   momentum=config.LOSS.MOMENTUM, scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)
    p2 = Prototype(num_pids=num_train_pids, feat_dim=config.MODEL.FEATURE_DIM, margin=0.3,
                    momentum=config.LOSS.MOMENTUM, scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)    
    for batch_idx, (ConditionSet, imgs, imgs_b, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        pos_mask = pos_mask.float().cuda()
        imgs, imgs_b, pids, camids, clothes_ids =  imgs.cuda(), imgs_b.cuda(), pids.cuda(), camids.cuda(), clothes_ids.cuda()
        for i in range(len(ConditionSet)):
            ConditionSet[i] = ConditionSet[i].cuda()
            
        data_time.update(time.time() - end)
        tuple_features = model(torch.cat((imgs, imgs_b), dim=0), mode=pids)
        features, features2 = tuple_features[1].split(imgs.size(0), dim=0)
        clo_feat = tuple_features[2]
        clo_score = tuple_features[3]
        mask_feat = tuple_features[4]
        clofeat = tuple_features[5]
        out1_2, out2_1, out1_1, out2_2 = genertor(features.detach(),clo_feat.detach(),ConditionSet[3],ConditionSet[4],ConditionSet[5],ConditionSet[0],ConditionSet[1],ConditionSet[2])
        outputs = classifier(features)
        outputs2 = classifier(features2)
        cla_loss = criterion_cla(outputs, pids) 
        black_loss = criterion_cla(outputs2, pids)
        CiclLoss = criterion_cicl(features, features2, pids) 
        DiscLoss = camadv_loss(Discriminator(features.detach(),ConditionSet[3],0), torch.ones(pids.size(0), 1).cuda()) + \
                    camadv_loss(Discriminator(clo_feat.detach(),ConditionSet[0],1), torch.ones(pids.size(0), 1).cuda()) 
        if epoch % 4 <= 1 :
            DiscLoss = DiscLoss +\
                    camadv_loss(Discriminator(out1_2.detach(),ConditionSet[3],0), torch.zeros(pids.size(0), 1).cuda()) + \
                    camadv_loss(Discriminator(out2_1.detach(),ConditionSet[0],1), torch.zeros(pids.size(0), 1).cuda()) 
        
        optimizer_cc.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(DiscLoss, optimizer_cc) as scaled_loss:
                scaled_loss.backward()
        else:
            DiscLoss.backward()
        optimizer_cc.step()
        Int_CeLoss =  criterion_cal(mask_feat, clothes_ids[:, 0]) + criterion_cla(clo_score,camids) + criterion_cal(clofeat, clothes_ids[:, 1]) 
        with torch.no_grad():  
            k = copy.deepcopy(outputs2).detach()
            p_feat = softmax(k)
            true_label_probs = p_feat[range(p_feat.shape[0]), pids]
        CiclLoss = torch.mean(CiclLoss * true_label_probs)
        _, preds = torch.max(outputs.data, 1)
        _, preds2 = torch.max(outputs2.data, 1)
        MMLoss = criterion_prototype(features, pids, p1, p2) + criterion_prototype(features2, pids, p2, p1)
        new_feat = classifier(features)-classifier(out2_1)
        TriLoss = criterion_tri(features2, pids) + criterion_tri(features, pids)
        GenLoss = camadv_loss(Discriminator(out2_1,ConditionSet[0],1), torch.ones(pids.size(0), 1).cuda()) + \
                   camadv_loss(Discriminator(out1_2,ConditionSet[3],0), torch.ones(pids.size(0), 1).cuda()) + \
                   CycleLoss(out1_1, features.detach()) + CycleLoss(out2_2, clo_feat.detach()) 

        loss = cla_loss +  black_loss + TriLoss + (CiclLoss+MMLoss)*config.Hyper.beta
        if epoch % 4 > 1:
            loss+= GenLoss+ Int_CeLoss*config.Hyper.alpha + criterion_cla(new_feat, pids)*config.Hyper.eta
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds2 == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_DiscLoss.update(DiscLoss.item(), pids.size(0))
        batch_TriLoss.update(TriLoss.item(), pids.size(0))
        batch_clo_loss.update(Int_CeLoss.item(), pids.size(0))
        batch_GenLoss.update(GenLoss.item(), pids.size(0))
        batch_CiclLoss.update(CiclLoss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'IdLoss:{DiscLoss.avg:.4f} '
                  'adv2Loss:{TriLoss.avg:.4f} '
                  'CloLoss:{Int_CeLoss.avg:.4f} '
                  'Fusloss:{GenLoss.avg:.4f} ' 
                  'TriLoss:{CiclLoss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, DiscLoss=batch_DiscLoss,
                   TriLoss=batch_TriLoss, 
                   Int_CeLoss=batch_clo_loss, 
                   GenLoss=batch_GenLoss, CiclLoss=batch_CiclLoss, 
                   acc=corrects, clo_acc=clothes_corrects))
    
def kl_loss(input1,input2):
    input1 = F.normalize(input1, p=2, dim=1)
    input2 = F.normalize(input2, p=2, dim=1)
    sim1 = torch.matmul(input1,input1.t())
    sim2 = torch.matmul(input2,input2.t())
    return F.kl_div(F.log_softmax(sim1,dim=1),F.softmax(sim2,dim=1),reduction='batchmean')+F.kl_div(F.log_softmax(sim2,dim=1),F.softmax(sim1,dim=1),reduction='batchmean')

def train_adv2(config, epoch, model, classifier, num_train_pids,criterion_cla, 
        criterion_prototype, criterion_cicl, criterion_tri, optimizer,trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_DiscLoss = AverageMeter()
    batch_TriLoss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_GenLoss = AverageMeter()
    batch_CiclLoss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    camadv_loss = nn.BCEWithLogitsLoss()
    CycleLoss = nn.MSELoss()
    model.train()
    classifier.train()

    end = time.time()
    softmax = nn.Softmax(dim=1)
    p1 = Prototype(num_pids=num_train_pids, feat_dim=config.MODEL.FEATURE_DIM, margin=0.3, 
                   momentum=config.LOSS.MOMENTUM, scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)
    p2 = Prototype(num_pids=num_train_pids, feat_dim=config.MODEL.FEATURE_DIM, margin=0.3,
                    momentum=config.LOSS.MOMENTUM, scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)    
    for batch_idx, (ConditionSet, imgs, imgs_b, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        pos_mask = pos_mask.float().cuda()
        imgs, imgs_b, pids=  imgs.cuda(), imgs_b.cuda(), pids.cuda()
        data_time.update(time.time() - end)
        tuple_features = model(torch.cat((imgs, imgs_b), dim=0), mode=pids)
        features, features2 = tuple_features[1].split(imgs.size(0), dim=0)
        outputs = classifier(features)
        outputs2 = classifier(features2)
        cla_loss = criterion_cla(outputs, pids) 
        black_loss = criterion_cla(outputs2, pids)
        CiclLoss = criterion_cicl(features, features2, pids) 
        with torch.no_grad():  
            k = copy.deepcopy(outputs2).detach()
            p_feat = softmax(k)
            true_label_probs = p_feat[range(p_feat.shape[0]), pids]
        CiclLoss = torch.mean(CiclLoss * true_label_probs)
        _, preds = torch.max(outputs.data, 1)
        _, preds2 = torch.max(outputs2.data, 1)
        MMLoss = criterion_prototype(features, pids, p1, p2) + criterion_prototype(features2, pids, p2, p1)
        TriLoss = criterion_tri(features2, pids) + criterion_tri(features, pids)
        loss = cla_loss +  black_loss + TriLoss + (CiclLoss+MMLoss)*config.Hyper.beta
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds2 == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_TriLoss.update(TriLoss.item(), pids.size(0))
        batch_CiclLoss.update(CiclLoss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'adv2Loss:{TriLoss.avg:.4f} '
                  'TriLoss:{CiclLoss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss,
                   TriLoss=batch_TriLoss, 
                   CiclLoss=batch_CiclLoss, 
                   acc=corrects, clo_acc=clothes_corrects))