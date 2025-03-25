import torch
import torch.nn.functional as F
from torch import nn
from losses.gather import GatherLayer
from losses.cross_entry_smooth import CrossEntropyWithLabelSmooth
class PrototypeBaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ranking_loss = CrossEntropyWithLabelSmooth()

    def forward(self, inputs, targets, prototype1, prototype2):
        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        prototype2.ema(gallery_inputs.detach(), gallery_targets)
        # l2-normlize
        inputs = F.normalize(inputs, p=2, dim=1)
        memory_norm = F.normalize(prototype1.feature_memory.detach(), p=2, dim=1).cuda()
        dist2 =  torch.matmul(inputs, memory_norm.t()) * prototype1.scale
        if not prototype1.has_been_filled:
            invalid_index = prototype1.label_memory == -1
            if sum(invalid_index.type(torch.int)) == 0:
                prototype1.has_been_filled = True
            else: return torch.tensor(0)
                
        loss = self.ranking_loss(dist2, targets)

        return loss
    