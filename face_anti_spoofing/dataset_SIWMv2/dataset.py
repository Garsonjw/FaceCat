# ***************************一些必要的包的调用********************************
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import cv2
import sys
sys.path.append('/home/kangcaixin/chenjiawei/ddpm-segmentation/face_anti_spoofing/')
from dataset_SIWMv2.config_siwm import Config_siwm
# ***************************初始化一些函数********************************

def default_loader(path):
    #return Image.open(path).convert('RGB')
    return cv2.imread(path)

class MyDataset(Dataset):
    def __init__(self, pro=1, unknown='None', adv_un='Facemask', train=True, txt_path='None', defense=False, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        self.loader = loader
        self.defense = defense
        config_siwm = Config_siwm(pro, unknown)
        config_siwm.compile()
        imgs = []
        path_save = []
        adv_train_dir = []
        adv_test_dir = []
        if train ==True:
            data_dir_li = config_siwm.LI_DATA_DIR
            data_dir_sp = config_siwm.SP_DATA_DIR
            data_dir_adv = config_siwm.file_reader('/home/kangcaixin/chenjiawei/ArcFace_256/adv_train.txt')
            if pro == 3:
                for x in data_dir_adv:
                    if adv_un not in x:
                        adv_train_dir.append(x)
                data_dir_adv = adv_train_dir
        else:
            data_dir_li = config_siwm.LI_DATA_DIR_TEST
            data_dir_sp = config_siwm.SP_DATA_DIR_TEST
            data_dir_adv = config_siwm.file_reader('/home/kangcaixin/chenjiawei/ArcFace_256/adv_test.txt')
            if pro == 3:
                for x in data_dir_adv:
                    if adv_un in x:
                        adv_test_dir.append(x)
                data_dir_adv = adv_test_dir

        for li in data_dir_li:
            for img_li in os.listdir(li):
                imgs.append((os.path.join(li, img_li), 1))  #1代表标签为1
                if train ==False:
                    path_save.append('{} {}\n'.format(os.path.join(li, img_li), 1))

        for sp in data_dir_sp:
            for img_sp in os.listdir(sp):
                imgs.append((os.path.join(sp, img_sp), 0))  #0代表标签为0
                if train ==False:
                    path_save.append('{} {}\n'.format(os.path.join(sp, img_sp), 0))

        for adv in data_dir_adv:
            imgs.append((adv, 0))  #0代表标签为0
            if train ==False:
                path_save.append('{} {}\n'.format(adv, 0))

        if train ==False:
            with open(txt_path +'/'+'test_path.txt', 'w') as file:
                file.writelines(path_save)

        self.imgs = imgs
        #print(imgs)
        self.transform = transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.defense ==True:
            #print('defense')
            img = torch.Tensor(img.transpose((2, 0, 1)))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    trainset = MyDataset(pro=1, train=False)
    print(len(trainset))
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset)-train_size
    print(train_size)
    print(val_size)
    train_set, val_set = random_split(trainset, lengths=[train_size, val_size], generator=torch.manual_seed(0))