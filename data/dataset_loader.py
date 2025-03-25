import copy
import os.path
# import cv2
import torch
import os.path as osp
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import data.img_transforms as T
import random
import numpy as np
import json
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_parsing_result(img_path):
    """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, mode="train", p=0.3, only_dsiflf = False):
        super(ImageDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.mode = mode
        self.pro = p
        self.od =  only_dsiflf
        if not self.od:
            self.text = torch.load('../caption/ltcc-tt_id.pt')
            self.clothes =  torch.load('../caption/ltcc-tt_int.pt')
            with open("../caption/ltcc-int.json", "r", encoding="utf-8") as file:
                self.indexpath = json.load(file)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        if self.mode == "test":
            img = read_image(img_path)
            img = self.transform(img)
            return img, pid, camid, clothes_id, img_path
        
       
        parsing_result_path = os.path.join('', '/'.join(img_path.split('/')[:-2]), 'processed',img_path.split('/')[-1][:-4] + '.png')
        img = read_image(img_path)
        parsing_result = read_parsing_result(parsing_result_path)

        parsing_result_copy = torch.tensor(np.asarray(parsing_result, dtype=np.uint8)).unsqueeze(0).repeat(3, 1, 1).detach()

        img_b = copy.deepcopy(img)
        img_b = np.array(img_b, dtype=np.uint8).transpose(2, 0, 1)

        target_classes = [2,3,4,5,6,7,8]
        p1 = random.randint(0, 1)
        p2 = random.randint(0, 1)
        p3 = random.randint(0, 1)

        transform_b = T.Compose([
            T.Resize((384, 192)),
            T.RandomCroping(p=p1),
            T.RandomHorizontalFlip(p=p2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=p3)
        ])
        probability = self.pro
        random_probabilities = np.random.rand(*parsing_result_copy.shape)
        img_b[np.isin(parsing_result_copy, target_classes) & (random_probabilities < probability)] = 0
        img = self.transform(img)
        img_b = img_b.transpose(1, 2, 0)
        img_b = Image.fromarray(img_b, mode='RGB')

        img_b = transform_b(img_b)
        if not self.od:
            id_features = self.text["id"+img_path].squeeze()
            id_features2 = self.text["ida"+img_path].squeeze()
            id_features3 = self.text["idt"+img_path].squeeze()
            clothes_features = self.clothes["tf"+img_path].squeeze()
            clothes_features2 = self.clothes["af"+img_path].squeeze()
            clothes_features3 = self.clothes["tt"+img_path].squeeze()
            # print(text_features.shape)
            result = (id_features,id_features2,id_features3,clothes_features,clothes_features2,clothes_features3)
            clothes_id = self.create_tensor(clothes_id,img_path)
            # print(result.shape)
            return result, img, img_b, pid, camid, clothes_id
        else :
            return img, img, img_b, pid, pid , pid
    def create_tensor(self,clothesid,img_path):
        values = [
            clothesid,
            int(self.indexpath[img_path])
        ]
        tensor = torch.tensor(values, dtype=torch.int64)  # 使用 float32 类型创建张量
        return tensor

class ImageDataset_test(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        super(ImageDataset_test, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return  img, pid, camid, clothes_id


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
