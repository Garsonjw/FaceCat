from __future__ import print_function, division
import torch
import argparse, os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision
from CDCNs import CDCNpp
from DeepPixBiS import DeepPixBiS
from CMFL import RGBDMH
from MCCNN_BCE_OCCL_GMM.MCCNNCenterLoss import MCCNNCenterLoss
from MCCNN_BCE_OCCL_GMM.losses import OCCL
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from dataset_SIWMv2.dataset import MyDataset

from utils import AvgrageMeter, my_metrics
from tqdm import tqdm
from FLIP.fas import flip_it, flip_v
from adv_detection.EST_net import Resnet18
# from adv_detection.DFRAA import dfraa
# from adv_detection.DFRAA_fet import DFRAA_fet
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet
from Depthnet import Depthnet
import timm
from FRT_PAD.models.pad_model import PA_Detector, Face_Related_Work, Cross_Modal_Adapter
from FRT_PAD.models.networks import PAD_Classifier
import sys
sys.path.append("/home/kangcaixin/chenjiawei/ddpm-segmentation")
from mae.models_mae import MaskedAutoencoderViT
from src.base_head import pixel_head
import swav
from swav.hubconf import resnet50
from functools import partial
from FRT_PAD.models.networks import Face_Recognition

def contrast_depth_conv(input):
    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


class Contrast_depth_loss(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss, self).__init__()
        return

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss().cuda()

        loss = criterion_MSE(contrast_out, contrast_label)
        # loss = torch.pow(contrast_out - contrast_label, 2)
        # loss = torch.mean(loss)

        return loss


saved_features = []

def save_hook(module, input, output):
    saved_features.append(output.detach())

# main function
def train_test():
    log = args.log + '_log/' + str(args.pro) + '/' + args.unknown
    isExists = os.path.exists(log)
    if not isExists:
        os.makedirs(log)
    log_file = open(log + '/' + 'log.txt', 'w')

    echo_batches = args.echo_batches

    print("SIWMv2\n ")

    log_file.write('SIWMv2\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?

    print('train from scratch!\n')
    log_file.write('train from scratch!\n')
    log_file.flush()
    if args.log == 'CDCN':
        model = CDCNpp()
    elif args.log == 'DeepPixBiS':
        model = DeepPixBiS()
    elif args.log == 'CMFL':
        model = RGBDMH()
    elif args.log == 'MCCNNCenterLoss':
        model = MCCNNCenterLoss()
    elif args.log == 'flip_it':
        model = flip_it()
    elif args.log == 'flip_v':
        model = flip_v()
    elif args.log == 'adv_EST':
        model = Resnet18()
    elif args.log == 'dfraa':
        model = dfraa(architecture='mlp')
        model_fet = DFRAA_fet().cuda()
    elif args.log == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid())
    elif args.log == "inceptionv3":
        model = torchvision.models.Inception3(init_weights=True)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid())
    elif args.log == "efficientnetb0":
        model = EfficientNet.from_name('efficientnet-b0')
        fc_features = model._fc.in_features
        model._fc = nn.Linear(fc_features, 1)
        model = nn.Sequential(
            model,
            nn.Sigmoid()
        )
    elif args.log == "vit-b-16":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        fc_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid())
    elif args.log == "swin-b":
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=1)
        model = nn.Sequential(
            model,
            nn.Sigmoid()
        )
    elif args.log == "depthnet":
        model = Depthnet()
    elif args.log == "frt":
        net_pad = PA_Detector()
        net_downstream = Face_Related_Work('Face_Recognition')
        net_adapter = Cross_Modal_Adapter(graph_type='Dense_Graph', batch_size=32)
        model = PAD_Classifier(net_pad, net_downstream, net_adapter, "FR")
        model = nn.Sequential(
            model,
            nn.Sigmoid()
        )
    elif args.log == "mae":
        mae_model = MaskedAutoencoderViT(
            img_size=256, patch_size=8, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True
        )
        checkpoint = torch.load("/home/kangcaixin/chenjiawei/ddpm-segmentation/checkpoints/mae/ffhq.pth", map_location='cpu')
        mae_model.load_state_dict(checkpoint['model'])
        mae_model = mae_model.cuda()

        hook_handles = []
        handle = mae_model.blocks[-12].register_forward_hook(save_hook)
        hook_handles.append(handle)

        model = pixel_head()
    elif args.log == "swav":
        swav_model = resnet50(pretrained=False).eval()
        swav_model.fc = nn.Identity()
        swav_model.cuda()
        state_dict = torch.load("/home/kangcaixin/chenjiawei/ddpm-segmentation/checkpoints/swav/ffhq.pth")['state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        swav_model.load_state_dict(state_dict, strict=False)
        hook_handles = []
        handle = swav_model.layer3.register_forward_hook(save_hook)
        hook_handles.append(handle)
        model = pixel_head()

    elif args.log == "vit-b-16-head":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.cuda()
        for param in model.parameters():
            param.requires_grad = False
        fc_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid())

    elif args.log == "face_re":
        net = Face_Recognition()
        net = net.cuda()
        model_path = './FRT_PAD/pretrained_model/backbone.pth'
        model_dict = net.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        hook_handles = []
        handle = net.layer4.register_forward_hook(save_hook)
        print(net)
        hook_handles.append(handle)
        model = pixel_head(dim=512)

    model = model.cuda()

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda()
    best_test_ACER = 1.0

    if args.log == 'flip_it' or args.log == 'flip_v':
        optimizer_dict = [
            {
                'params': filter(lambda p: p.requires_grad, model.parameters()),
                'lr': 0.000001
            },
        ]
        optimizer = optim.Adam(optimizer_dict, lr=0.000001, weight_decay=0.000001)
        criterion = {'softmax': nn.CrossEntropyLoss().cuda()}
    if args.log == 'adv_EST':
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.6)

    #####zk tensor
    #num_elements = int(12.5e9)  # 这是一个非常大的数字
    #large_tensor = torch.zeros(num_elements, dtype=torch.float32, device='cuda')
    #print(large_tensor.size())

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        # scheduler.step() #注意scheduler的位置
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_total = AvgrageMeter()

        ###########################################
        '''                train             '''
        ###########################################
        model.train()

        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomErasing(), transforms.RandomHorizontalFlip(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])])
        if args.log == 'MCCNNCenterLoss':
            train_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.RandomErasing(), transforms.RandomHorizontalFlip(),transforms.Resize((128, 128)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((128, 128)),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],   std=[0.229, 0.224, 0.225])])

        if args.log == 'flip_it' or args.log == 'flip_v' or args.log == 'vit-b-16' or args.log == "swin-b" or args.log == 'vit-b-16-head':
            train_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.RandomErasing(), transforms.RandomHorizontalFlip(),transforms.Resize((224, 224)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224)),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],   std=[0.229, 0.224, 0.225])])

        if args.log == 'inceptionv3':
            train_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.RandomErasing(), transforms.RandomHorizontalFlip(),transforms.Resize((299, 299)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((299, 299)),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],   std=[0.229, 0.224, 0.225])])


        train_data = MyDataset(pro=args.pro, train=True, unknown=args.unknown, transform=train_transforms)
        test_data = MyDataset(pro=args.pro, train=False, txt_path=log, unknown=args.unknown,
                              transform=test_transforms)
        train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=8)

        if args.log == 'frt':
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8, drop_last=True)

        i = 0
        for images, lables in tqdm(train_loader):
            i = i + 1
            # get the inputs
            images, lables = images.cuda(), lables.cuda()
            optimizer.zero_grad()
            # forward + backward + optimize
            if args.log == 'CDCN':
                map_x, _, _, _, _, _ = model(images)

                map_label = []
                for X in range(len(lables)):
                    if lables[X] == 0:
                        map_label.append(np.zeros([32, 32]))
                    else:
                        map_label.append(np.ones([32, 32]))

                map_label = torch.tensor(np.array(map_label), dtype=torch.float32).cuda()

                absolute_loss = criterion_absolute_loss(map_x, map_label)
                contrastive_loss = criterion_contrastive_loss(map_x, map_label)

                loss = absolute_loss + contrastive_loss

            elif args.log == 'mae':
                with torch.no_grad():
                    saved_features.clear()
                    _, _, ids_restore = mae_model.forward_encoder(images, mask_ratio=0)
                    ids_restore = ids_restore.unsqueeze(-1)
                    sqrt_num_patches = int(mae_model.patch_embed.num_patches ** 0.5)
                    for feat in saved_features:
                        feat = feat[:, 1:]
                        feat = torch.gather(feat, dim=1, index=ids_restore.repeat(1, 1, feat.shape[2]))
                        feat = feat.permute(0, 2, 1)
                        feat = feat.view(*feat.shape[:2], sqrt_num_patches, sqrt_num_patches)
                score = model(feat)
                loss = nn.BCELoss()(score.squeeze(-1).float(), lables.float())

            elif args.log == 'swav':
                with torch.no_grad():
                    saved_features.clear()
                    swav_model(images)
                    feat = saved_features[0]
                # for handle in hook_handles:
                #     handle.remove()
                score = model(feat)
                loss = nn.BCELoss()(score.squeeze(-1).float(), lables.float())

            elif args.log == "vit-b-16-head":
                score = model(images)
                loss = nn.BCELoss()(score.squeeze(-1).float(), lables.float())

            elif args.log == "face_re":
                with torch.no_grad():
                    saved_features.clear()
                    images = F.interpolate(images, (112, 112), mode='bilinear', align_corners=True)
                    net(images)
                    feat = saved_features[0]
                score = model(feat)
                loss = nn.BCELoss()(score.squeeze(-1).float(), lables.float())

            elif args.log == 'DeepPixBiS':
                map_x, score = model(images)
                map_label = []
                for X in range(len(lables)):
                    if lables[X] == 0:
                        map_label.append(np.zeros([16, 16]))
                    else:
                        map_label.append(np.ones([16, 16]))

                map_label = torch.tensor(np.array(map_label), dtype=torch.float32).cuda()
                loss = 0.5 * criterion_absolute_loss(map_x.squeeze(1), map_label) + 0.5 * nn.BCELoss()(
                    score.squeeze(-1).float(), lables.float())

            elif args.log=='CMFL' or args.log=="resnet50" or args.log=="inceptionv3" or args.log=="efficientnetb0" or args.log=="vit-b-16" or args.log =="swin-b" or args.log =='frt':
                if args.log == "inceptionv3":
                    score, _ = model(images)
                else:
                    score = model(images)
                    # print(score)
                loss = nn.BCELoss()(score.squeeze(-1).float(), lables.float())

            elif args.log == 'MCCNNCenterLoss':
                embedding, output = model(images)
                criterion = nn.BCELoss()
                criterion_center = OCCL(margin=3.0, feat_dim=10, alpha=0.1)
                loss = 0.5 * criterion(output.squeeze(-1), lables.float()) + 0.5 * criterion_center(embedding,
                                                                                                    lables.float(),
                                                                                                    model.training)
            elif args.log == 'flip_it' or args.log == 'flip_v':
                classifier_label_out, feature = model(images, True)
                loss = criterion['softmax'](classifier_label_out.narrow(0, 0, images.size(0)), lables)

            elif args.log == 'adv_EST':
                map_label = []

                for X in range(len(lables)):
                    if lables[X] == 0:
                        map_label.append(np.zeros([8, 8]))
                    else:
                        map_label.append(np.ones([8, 8]))


                map_label = torch.tensor(np.array(map_label), dtype=torch.float32).cuda()
                map_x = model(images)
                scores = torch.clamp(torch.mean(map_x, dim=(1, 2)), 0, 1)
                # print(scores)
                # print(soi)
                loss = nn.MSELoss()(map_x.squeeze(1), map_label)

            elif args.log == 'dfraa':
                model_fet.eval()
                features = model_fet(images)
                score = model(features)
                loss = nn.BCELoss()(score.squeeze(-1).float(), lables.float())

            elif args.log == "depthnet":
                map_label = []

                for X in range(len(lables)):
                    if lables[X] == 0:
                        map_label.append(np.zeros([32, 32]))
                    else:
                        map_label.append(np.ones([32, 32]))

                map_label = torch.tensor(np.array(map_label), dtype=torch.float32).cuda()
                map_x, score = model(images)
                # scores = torch.clamp(torch.mean(map_x, dim=(1, 2)), 0, 1)
                loss = nn.MSELoss()(map_x.squeeze(1), map_label) + nn.BCELoss()(score.squeeze(-1).float(), lables.float())

            loss.backward()
            optimizer.step()

            # loss_feature.update(feature_loss.data, images.size(0))
            loss_total.update(loss.data, images.size(0))

            if i % echo_batches == echo_batches - 1:  # print every 200 mini-batches

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, lable_loss= %.4f\n' % (epoch + 1, i + 1, lr, loss_total.avg))

        # whole epoch average
        print('epoch:%d, Train: lables_loss= %.4f\n' % (epoch + 1, loss_total.avg))
        log_file.write('epoch:%d, Train: lables_loss= %.4f \n' % (epoch + 1, loss_total.avg))
        log_file.flush()

        #### validation/test
        if epoch < 100:
            epoch_test = 1
        else:
            epoch_test = 10

        if epoch % epoch_test == epoch_test - 1:  # test every 5 epochs
            model.eval()

            with torch.no_grad():
                ###########################################
                '''                test             '''
                ##########################################

                for images, lables in tqdm(test_loader):
                    # get the inputs
                    images, lables = images.cuda(), lables.cuda()
                    optimizer.zero_grad()

                    if args.log == 'CDCN':
                        map_x, _, _, _, _, _ = model(images)
                        scores = torch.clamp(torch.mean(map_x, dim=(1, 2)), 0, 1)
                    elif args.log == 'DeepPixBiS':
                        map_x, score = model(images)
                        scores = (torch.clamp(torch.mean(map_x.squeeze(1), dim=(1, 2)), 0, 1) + score.squeeze(-1)) / 2
                    elif args.log == 'CMFL' or args.log == "resnet50" or args.log == "inceptionv3" or args.log == "efficientnetb0" or args.log == "vit-b-16" or args.log=="swin-b" or args.log=='frt' or args.log == 'vit-b-16-head':
                        scores = model(images)
                    elif args.log == 'MCCNNCenterLoss':
                        _, scores = model(images)
                    elif args.log == 'flip_it' or args.log == 'flip_v':
                        cls_out, feature = model(images, True)
                        scores = F.softmax(cls_out, dim=1)[:, 1]
                    elif args.log == 'adv_EST':
                        map_x = model(images)
                        scores = torch.clamp(torch.mean(map_x, dim=(1, 2)), 0, 1)
                    elif args.log == 'depthnet':
                        map_x, score =model(images)
                        scores = (torch.clamp(torch.mean(map_x.squeeze(1), dim=(1, 2)), 0, 1) + score.squeeze(-1)) / 2
                    elif args.log == 'mae':
                        with torch.no_grad():
                            saved_features.clear()
                            _, _, ids_restore = mae_model.forward_encoder(images, mask_ratio=0)
                            ids_restore = ids_restore.unsqueeze(-1)
                            sqrt_num_patches = int(mae_model.patch_embed.num_patches ** 0.5)
                            for feat in saved_features:
                                feat = feat[:, 1:]
                                feat = torch.gather(feat, dim=1, index=ids_restore.repeat(1, 1, feat.shape[2]))
                                feat = feat.permute(0, 2, 1)
                                feat = feat.view(*feat.shape[:2], sqrt_num_patches, sqrt_num_patches)
                        scores = model(feat)
                    elif args.log == 'swav':
                        with torch.no_grad():
                            saved_features.clear()
                            swav_model(images)
                            feat = saved_features[0]
                        # for handle in hook_handles:
                        #     handle.remove()
                        scores = model(feat)
                    elif args.log == 'face_re':
                        with torch.no_grad():
                            images = F.interpolate(images, (112, 112), mode='bilinear', align_corners=True)
                            saved_features.clear()
                            net(images)
                            feat = saved_features[0]
                        scores = model(feat)


                    score_test_filename = log + '/' + 'epoch_' + str(epoch + 1) + '_score_test.txt'
                    with open(score_test_filename, 'a') as f:
                        for score, lable in zip(scores, lables):
                            f.writelines('{} {}\n'.format(score.item(), lable))

                #############################################################
                #       performance measurement both val and test
                #############################################################
                test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l = my_metrics(score_test_filename)

                if test_ACER < best_test_ACER:

                    path = log + '/checkpoint/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    best_test_ACER = test_ACER
                    torch.save(model.state_dict(), path + args.log + '_test_Best(epoch=' + str(epoch + 1) + ')' + '.pt')
                    log_file.write("Fas model saved to {}\n".format(path + args.log + '_test_Best(epoch=' + str(epoch + 1) + ')' + '.pt'))
                    print("[Best result] epoch:{}, test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}\n".format(
                        epoch + 1, test_APCER, test_BPCER, test_ACER))

                log_file.write(
                    "epoch:{:d}, test:  test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}, test_EER={:.4f}, res_tpr={:.4f}, auc_score={:.4f}, tpr_h={:.4f}, tpr_m={:.4f}, tpr_l={:.4f}\n".format(
                        epoch + 1, test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m,
                        tpr_l))
                print(
                    "epoch:{:d}, test:  test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}, test_EER={:.4f}, res_tpr={:.4f}, auc_score={:.4f}, tpr_h={:.4f}, tpr_m={:.4f}, tpr_l={:.4f}\n".format(
                        epoch + 1,
                        test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l))

                log_file.flush()

    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  # lr 0.0001应该也可以
    parser.add_argument('--batchsize', type=int, default=64, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=50, help='how many epochs lr decays once')  # 500
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=200, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--log', type=str, default="face_re", help='["CDCN","DeepPixBiS","CMFL","MCCNNCenterLoss", "flip_it", "flip_v", "adv_EST", "dfraa","resnet50", "inceptionv3", "efficientnetb0", "vit-b-16", "swin-b", "depthnet","frt","mae","swav","vit-b-16-head","face_re"]')
    parser.add_argument('--pro', type=int, default=1, help='protocol: 1 or 2')
    parser.add_argument('--unknown', type=str, default='None',
                        help="{'None','Co', 'Im', 'Ob','Half', 'Mann', 'Paper','Sil', 'Trans', 'Print','Eye', 'Funnyeye','Mouth', 'Paperglass','Replay'}")

    args = parser.parse_args()
    global features_state
    features_state = OrderedDict()
    train_test()