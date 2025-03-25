import time
import logging
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes


def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs


@torch.no_grad()
def extract_img_feature(model, dataloader):
    features,features_b, pids, camids, clothes_ids, preds = [],[], torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    batch_path = []
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, batch_img_path) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        # imgs, imgs_b, filp_imgs2, flip_imgs = imgs.cuda(), imgs_b.cuda(), filp_imgs2.cuda(), flip_imgs.cuda()
        imgs,flip_imgs=imgs.cuda(),flip_imgs.cuda()
        pred, batch_features = model(imgs,mode='test')
        _, batch_features_flip = model(flip_imgs,mode='test')
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)
        batch_path+=batch_img_path
        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
        preds = torch.cat((preds, pred.cpu()), dim=0)
    features = torch.cat(features, 0)
    return features, pids, camids, clothes_ids, preds


def test(config, model,queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    
    local_rank = dist.get_rank()
    # Extract features 
    qf, q_pids, q_camids, q_clothes_ids,q_preds = extract_img_feature(model, queryloader)
    gf, g_pids, g_camids, g_clothes_ids,g_preds = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qf, q_pids, q_camids, q_clothes_ids,q_preds = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids,q_preds], len(dataset.query))
    gf, g_pids, g_camids, g_clothes_ids,g_preds = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids,g_preds], len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    q_preds_2d = q_preds.unsqueeze(1)  
    g_preds_2d = g_preds.unsqueeze(0)  
    comparison = q_preds_2d == g_preds_2d
    result_tensor = comparison.int()
    distmat = distmat + result_tensor*config.Hyper.k
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_prcc(config, model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids,qs_preds= extract_img_feature(model, None, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids,qd_preds= extract_img_feature(model, None, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids,g_preds = extract_img_feature(model, None, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids,qs_preds = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids,qs_preds], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids,qd_preds = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids,qd_preds], len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids,g_preds = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids,g_preds], len(dataset.gallery))
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # cf = k_means(gf,80)
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu() 
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu() 
    # qs_preds_2d = qs_preds.unsqueeze(1)  
    qd_preds_2d = qd_preds.unsqueeze(1) 
    g_preds_2d = g_preds.unsqueeze(0)  
    # comparison = qs_preds_2d == g_preds_2d
    comparison2 = qd_preds_2d == g_preds_2d
    print((torch.sum(qs_preds == qs_camids.data)+torch.sum(qd_preds == qd_camids.data)+torch.sum(g_preds == g_camids.data))/(m+n+k))
    # result_tensor = comparison.int()
    result_tensor2 = comparison2.int()
    distmat_same = distmat_same
    distmat_diff = distmat_diff + config.Hyper.k*result_tensor2
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]

def test2(config, model,queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    
    local_rank = dist.get_rank()
    # Extract features 
    qf, q_pids, q_camids, q_clothes_ids,_ = extract_img_feature(model, queryloader)
    gf, g_pids, g_camids, g_clothes_ids,_ = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
 
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_prcc2(config, model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids,_= extract_img_feature(model, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids,_= extract_img_feature(model, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids,_ = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu() 
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu() 
    # qs_preds_2d = qs_preds.unsqueeze(1)  
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]