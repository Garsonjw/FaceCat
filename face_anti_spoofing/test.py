
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/kangcaixin/chenjiawei/ddpm-segmentation")
from src.utils import setup_seed
from src.pixel_classifier import pixel_classifier
from face_anti_spoofing.utils import my_metrics
from src.feature_extractors import create_feature_extractor, f16to32
from SAFAS.utils import *
from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev
import argparse
from prompt_templates import spoof_templates, real_templates
import torchvision.transforms as transforms
from face_anti_spoofing.dataset_SIWMv2.dataset import MyDataset
import clip
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from openTSNE import TSNE
import pandas as pd
import seaborn as sns
import random
import matplotlib.patches as patches
from defense_transform import DiffJPEG, DiffBitReduction, DiffRandomization
from CDCNs import CDCNpp
from CMFL import RGBDMH
from Depthnet import Depthnet
def generate_random_colors(num_colors=29):
    colors = []

    for _ in range(num_colors):
        colors.append('#{:06x}'.format(random.randint(0, 0xFFFFFF)))

    return colors

def test(args):
    log = args.log + '_log/' + str(args.pro)
    isExists = os.path.exists(log)
    if not isExists:
        os.makedirs(log)
    with torch.no_grad():
        if args.log == 'test_CDCN_oppo':
            classifier = CDCNpp().cuda()
            classifier_path = "/face_anti_spoofing/checkpoints/CDCN_test_Best(epoch=20).pt"
        elif args.log == 'test_CMFL_oppo':
            classifier = RGBDMH().cuda()
            classifier_path = "/face_anti_spoofing/checkpoints/CMFL_test_Best(epoch=11).pt"
        elif args.log == 'test_depthnet_oppo':
            classifier = Depthnet().cuda()
            classifier_path = "/face_anti_spoofing/checkpoints/depthnet_test_Best(epoch=16).pt"

        classifier.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(classifier_path).items()})
        classifier.eval()

        test_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_data = MyDataset(pro=args.pro, train=False, txt_path=log, defense=False, transform=test_transforms)
        # test_data = MyDataset(txt='/home/kangcaixin/chenjiawei/physical_256/path_1.txt', transform=test_transforms)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
        feats = []
        ls = []
        scores_list = []
        for images, labels in tqdm(test_loader):
            images, labels = images.to(dev()), labels.to(dev())
            if args.log == 'test_CDCN_oppo':
                map_x, _, _, _, _, _ = classifier(images)
                scores = torch.clamp(torch.mean(map_x, dim=(1, 2)), 0, 1)
            elif args.log == 'test_CMFL_oppo':
                scores = classifier(images)
            elif args.log == 'test_depthnet_oppo':
                map_x, score = classifier(images)
                scores = (torch.clamp(torch.mean(map_x.squeeze(1), dim=(1, 2)), 0, 1) + score.squeeze(-1)) / 2
            for j in range(len(scores)):
                scores_list.append('{} {}\n'.format(scores[j].item(), labels[j].item()))
                
    map_score_test_filename = log + '/'+'defense_score_test.txt'

    with open(map_score_test_filename, 'w') as file:
        file.writelines(scores_list)
    print("Finish test")

    test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l = my_metrics(
        map_score_test_filename)
    print("test:  test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}, test_EER={:.4f}, res_tpr={:.4f}, auc_score={:.4f}, tpr_h={:.4f}, tpr_m={:.4f}, tpr_l={:.4f}\n".format(test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--clip', action='store_true', help="used clip_loss?")
    parser.add_argument("--log", default='test_depthnet_oppo', type=str, help="test_CDCN_oppo, test_CMFL_oppo, test_depthnet_oppo")
    parser.add_argument('--pro', type=int, default=1, help='protocol: 1 or 2 or 3')
    parser.add_argument('--batch_size', type=int, default=64, help='initial batchsize')

    args = parser.parse_args()
    setup_seed(args.seed)

    test(args)
#CUDA_VISIBLE_DEVICES='4' python test.py
