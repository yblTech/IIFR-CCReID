import torch
import torch.nn as nn
from torch.nn import Module
from torch import tensor
import copy
# from torchvision.models import resnet50, ResNet50_Weights
import timm
from torch.nn import init
import torchvision
from models.rev import revgrad
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class MaxAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)

        return torch.cat((max_f, avg_f), 1)


class RevGrad(Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)
class build_transformer_vit_local_grl(nn.Module):
    def __init__(self, num_classes, num_attributes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_vit_local_grl, self).__init__()
        model_path = '/home/ybli/lyb/AIM/jx_vit_base_p16_224-80ecf9dd.pth'
        pretrain_choice = 'imagenet'
        self.cos_layer = False
        self.neck = 'bnneck'
        self.neck_feat = 'before'
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format('vit_base_patch16_224_TransReID'))

        if False:
            camera_num = camera_num
        else:
            camera_num = 0

        if False:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory['vit_base_patch16_224_TransReID'](img_size=[384,192], sie_xishu=3.0, local_feature=True, camera=camera_num, view=view_num, stride_size=[16, 16], drop_path_rate=0.1)

        if 'vit_base_patch16_224_TransReID' == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm

        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.b3 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.ID_LOSS_TYPE = 'softmax'

    
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_grl = nn.Sequential(    # Gradient Reversal Layer (GRL)
                RevGrad(),
                nn.Linear(self.in_planes, 384, bias=False),
                nn.BatchNorm1d(384),
                nn.GELU(),
                nn.Dropout2d(),
                nn.Linear(384, 96, bias=False),
                nn.BatchNorm1d(96),
                nn.GELU(),
                nn.Dropout2d(),
                nn.Linear(96, 48, bias=False),
                nn.BatchNorm1d(48),
                nn.GELU(),
                nn.Dropout2d(),
                nn.Linear(48, self.num_attributes, bias=False)
            )
        self.classifier_grl.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.bottleneck_grl = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_grl.bias.requires_grad_(False)
        self.bottleneck_grl.apply(weights_init_kaiming)

        self.shuffle_groups = 2
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = 5
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = 4
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        # grl attribute branch
        b3_feat = self.b3(features)
        attr_feat = b3_feat[:, 0] 

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        attr_feat_bn = self.bottleneck_grl(attr_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
                cls_score_attr = self.classifier_grl(attr_feat_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4, cls_score_attr
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4, attr_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4, attr_feat_bn], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4, attr_feat], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i.replace('module.', '') in self.state_dict():
                try:
                    self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
                except RuntimeError as e:
                    if "The size of tensor a (96) must match the size of tensor b (48) at non-singleton dimension 1" in str(e):
                        print("忽略特定的大小不匹配错误")
            else:
                print(f"Warning: Key 'module.{i}' not found in the model state_dict.")
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))