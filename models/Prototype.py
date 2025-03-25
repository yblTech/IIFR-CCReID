import torch
import torch.nn.functional as F
from torch import nn

class Prototype(nn.Module):
    def __init__(self, num_pids, feat_dim, margin, momentum, scale, epsilon):
        super().__init__()
        self.m = margin
        self.num_pids = num_pids
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale

        self.register_buffer('prototype_feature', torch.zeros((num_pids, feat_dim)))
        self.register_buffer('label', torch.zeros(num_pids, dtype=torch.int64) - 1)
        self.has_been_filled = False

    
    def ema(self, features, labels):
        label_to_feat = {}
        for x, y in zip(features, labels):
            if y not in label_to_feat:
                label_to_feat[y] = [x.unsqueeze(0)]
            else:
                label_to_feat[y].append(x.unsqueeze(0))
        if not self.has_been_filled:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0).to(self.prototype_feature[y].device)
                self.prototype_feature[y] = feat
                self.label[y] = y
        else:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0).to(self.prototype_feature[y].device)
                self.prototype_feature[y] = self.momentum * self.prototype_feature[y] + (1. - self.momentum) * feat