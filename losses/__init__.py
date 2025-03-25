from torch import nn
from losses.cross_entry_smooth import CrossEntropyWithLabelSmooth
from losses.cosface import CosFaceLoss
from losses.prototype_base_loss import PrototypeBaseLoss
from losses.triplet_loss import TripletLoss, CICLLoss

def build_losses(config, num_train_clothes, num_train_pids):
    # Build identity classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))
    criterion_pair = TripletLoss(margin=0.3)
    criterion_prototype = PrototypeBaseLoss()
    # Build clothes-based adversarial loss

    if config.LOSS.CAL == 'cal':
        criterion_cicl = CICLLoss(margin=0.3)
        criterion_cal = CosFaceLoss(scale=config.LOSS.CLA_S, margin=0)
    else:
        raise KeyError("Invalid clothing classification loss: '{}'".format(config.LOSS.CAL))

    return criterion_cla, criterion_prototype, criterion_cal, criterion_cicl, criterion_pair
