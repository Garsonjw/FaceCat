from tqdm import tqdm
import json
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions, \
    pixel_classifier
from src.feature_extractors import create_feature_extractor, collect_features, f16to32

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev
from SAFAS.utils import *
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from src.pub_mod import *
from face_anti_spoofing.dataset_SIWMv2.dataset import MyDataset
from face_anti_spoofing.utils import my_metrics
from torchvision import transforms
from triplet_loss import TripletLoss
from focal_loss import BCEFocalLoss
import clip
from prompt_templates import spoof_templates, real_templates
import torch.nn.functional as F

# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434

def train(args):

    log = args["log"] + '_log/' + str(args["pro"]) + '/' + args["unknown"]
    if args["pro"] == 3:
        log = args["log"] + '_log/' + str(args["pro"]) + '/' + args["adv_un"]

    isExists = os.path.exists(log)
    if not isExists:
        os.makedirs(log)
    log_file = open(log + '/' + 'log.txt', 'w')
    best_test_ACER = 1.0
    # 多卡
    if args['muti_gpus'] == True:
        torch.cuda.set_device(args["local_rank"])
        dist.init_process_group(backend='nccl')  # nccl  # nccl是GPU设备上最快、最推荐的后端

    feature_extractor = create_feature_extractor(**args)
    print(f"Preparing the train set for {args['category']}...")

    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomErasing(), transforms.RandomHorizontalFlip(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])])

    train_data = MyDataset(pro=args["pro"], train=True, unknown=args["unknown"], adv_un=args["adv_un"], transform=train_transforms)
    test_data = MyDataset(pro=args["pro"], train=False, txt_path=log, unknown=args["unknown"], adv_un=args["adv_un"], transform=test_transforms)
    # 多卡
    if args['muti_gpus'] == True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

        train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=False, sampler=train_sampler, num_workers=8)
        test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False, sampler=test_sampler, num_workers=8)
        classifier = pixel_classifier(dim=args['dim'][-1]).to(args["local_rank"])
    else:
        train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=8)
        test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False, num_workers=8)
        classifier = pixel_classifier().cuda()
    epochs = 30
    # 多卡
    if args['muti_gpus'] == True:
        classifier = DDP(classifier, device_ids=[args["local_rank"]], output_device=args["local_rank"], find_unused_parameters=True)

    # criterion = nn.BCELoss()
    criterion_focal = BCEFocalLoss()
    criterion_trip = TripletLoss(device=torch.device('cuda'))
    # criterion_mse = nn.MSELoss()
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    if args['clip']:
        clip_model, _ = clip.load("ViT-B/16", 'cuda:0')
        clip_model.eval()
        with torch.no_grad():
            spoof_texts = clip.tokenize(spoof_templates).cuda(non_blocking=True)  # tokenize
            real_texts = clip.tokenize(real_templates).cuda(non_blocking=True)  # tokenize
            # embed with text encoder
            spoof_class_embeddings = clip_model.encode_text(spoof_texts)
            spoof_class_embeddings = spoof_class_embeddings.mean(dim=0)
            real_class_embeddings = clip_model.encode_text(real_texts)
            real_class_embeddings = real_class_embeddings.mean(dim=0)

            ensemble_weights = [spoof_class_embeddings, real_class_embeddings]
            text_features = torch.stack(ensemble_weights, dim=0).cuda()

    for epoch in range(epochs):
        loss_all = AvgrageMeter()
        # Contra_record = AvgrageMeter()
        B_L_record = AvgrageMeter()
        Trip_L_record = AvgrageMeter()
        ########################### train ###########################
        classifier.train()
        # 多卡
        if args['muti_gpus'] == True:
            train_loader.sampler.set_epoch(epoch)
        i = 0
        for batch_images, batch_labels in tqdm(train_loader):
            i = i + 1
            lr = optimizer.param_groups[0]['lr']

            batch_images, batch_labels = batch_images.to(dev()), batch_labels.to(dev())

            if 'share_noise' in args and args['share_noise']:
                rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
                noise = torch.randn(len(batch_images), 3, args['image_size'], args['image_size'],
                                    generator=rnd_gen, device=dev())
            else:
                noise = None  # 我修改到for循环里面了，这样noise的size可以动态变化

            features = feature_extractor(batch_images, noise=noise)
            features = f16to32(features)
            labels = batch_labels  # 200
            optimizer.zero_grad()

            scores, feat = classifier(features)
            # B_L = criterion(scores.squeeze(-1).cuda(), labels.float().cuda())
            if args['clip']:
                feat = feat / feat.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.to(torch.float32)
                logit_scale = clip_model.logit_scale.exp()
                logits_per_image = logit_scale * feat @ text_features.t()

                similarity = logits_per_image
                scores = F.softmax(similarity, dim=1)[:, 1]
                B_L = criterion_focal(scores.squeeze(-1).cuda(), labels.float().cuda())
            else:
                B_L = criterion_focal(scores.squeeze(-1).cuda(), labels.float().cuda())
            Trip_L = criterion_trip(feat, labels.float())
            # Trip_L = torch.tensor(0)
            loss = B_L + 0.1*Trip_L
            # loss = Trip_L
            loss.backward()
            optimizer.step()

            # Contra_record.update(contrast_loss.data, cls_x1_x1.size(0))
            B_L_record.update(B_L.data, scores.size(0))
            Trip_L_record.update(Trip_L.data, scores.size(0))
            loss_all.update(loss.data, scores.size(0))
            log_info = "epoch:{:d}, mini-batch:{:d}, lr={:.4f}, loss_all={:.4f}, bce_loss={:4f}, , trip_loss={:4f}\n".format(epoch + 1, i + 1, lr,
                                                                                        loss_all.avg, B_L_record.avg, Trip_L_record.avg)
            if i % args['print_freq'] == args['print_freq'] - 1:
                if args['muti_gpus'] == True:
                    if dist.get_rank() == 0:
                        print(log_info)
                        log_file.write(log_info)
                        log_file.flush()
                else:
                    print(log_info)
                    log_file.write(log_info)
                    log_file.flush()

        if args['muti_gpus'] == True:
            if dist.get_rank() == 0:
                print("epoch:{:d}, Train: lr={:f}, loss_all={:.4f}, bce_loss={:4f}, trip_loss={:4f}\n".format(epoch + 1, lr, loss_all.avg, B_L_record.avg, Trip_L_record.avg))
                log_file.write("epoch:{:d}, Train: lr={:f}, loss_all={:.4f}, bce_loss={:4f}, trip_loss={:4f}\n".format(epoch + 1, lr, loss_all.avg, B_L_record.avg, Trip_L_record.avg))
                log_file.flush()
        else:
            print("epoch:{:d}, Train: lr={:f}, loss_all={:.4f}, bce_loss={:4f}, trip_loss={:4f}\n".format(epoch + 1, lr, loss_all.avg, B_L_record.avg, Trip_L_record.avg))
            log_file.write(("epoch:{:d}, Train: lr={:f}, loss_all={:.4f}, bce_loss={:4f}, trip_loss={:4f}\n".format(epoch + 1, lr, loss_all.avg, B_L_record.avg, Trip_L_record.avg)))
            log_file.flush()
        scheduler.step()


        ############################ test ###########################
        if epoch < 100:
            epoch_test = 1
        if epoch % epoch_test == epoch_test - 1:
            classifier.eval()
            with torch.no_grad():
                scores_list = []
                for images, labels in tqdm(test_loader):
                    images, labels = images.to(dev()), labels.to(dev())
                    if 'share_noise' in args and args['share_noise']:
                        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
                        noise = torch.randn(len(images), 3, args['image_size'], args['image_size'],
                                            generator=rnd_gen, device=dev())
                    else:
                        noise = None

                    features = feature_extractor(images, noise=noise)
                    features = f16to32(features)  # [25, 8448, 256, 256]
                    scores, feat = classifier(features)
                    if args['clip']:
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        text_features = text_features.to(torch.float32)
                        logit_scale = clip_model.logit_scale.exp()
                        logits_per_image = logit_scale * feat @ text_features.t()

                        similarity = logits_per_image
                        scores = F.softmax(similarity, dim=1)[:, 1]

                    scores = scores.squeeze(-1)

                    if args['muti_gpus'] == True:
                        combined_tensor = torch.cat((scores, labels), dim=0)
                        gpus_num = dist.get_world_size()
                        combined_list = [torch.zeros(len(combined_tensor), dtype=torch.float32).cuda() for _ in
                                         range(gpus_num)]
                        dist.all_gather(combined_list, combined_tensor)

                        for i in range(len(combined_list)):
                            for j in range(len(images)):
                                scores_list.append(
                                    "{} {}\n".format(combined_list[i][j].item(), combined_list[i][j + len(images)].item()))
                    else:
                        for j in range(len(scores)):
                            scores_list.append('{} {}\n'.format(scores[j].item(), labels[j].item()))
            map_score_test_filename = log + '/' + 'epoch_' + str(epoch + 1) + '_score_test.txt'
            if args['muti_gpus'] == True:
                if dist.get_rank() == 0:
                    print("score: write test scores to {}".format(map_score_test_filename))
                    log_file.write("score: write test scores to {}\n".format(map_score_test_filename))
                    log_file.flush()
            else:
                print("score: write test scores to {}".format(map_score_test_filename))
                log_file.write("score: write test scores to {}\n".format(map_score_test_filename))
                log_file.flush()

            with open(map_score_test_filename, 'w') as file:
                file.writelines(scores_list)
                if args['muti_gpus'] == True:
                    dist.barrier()
            print("Finish test")

            test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l = my_metrics(map_score_test_filename)
            if args['muti_gpus'] == True:
                if dist.get_rank() == 0:
                    log_file.write("epoch:{:d}, test:  test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}, test_EER={:.4f}, res_tpr={:.4f}, auc_score={:.4f}, tpr_h={:.4f}, tpr_m={:.4f}, tpr_l={:.4f}\n".format(
                        epoch + 1, test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l))
                    log_file.flush()
                    print("epoch:{:d}, test:  test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}, test_EER={:.4f}, res_tpr={:.4f}, auc_score={:.4f}, tpr_h={:.4f}, tpr_m={:.4f}, tpr_l={:.4f}\n".format(
                        epoch + 1, test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l))
            else:
                log_file.write("epoch:{:d}, test:  test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}, test_EER={:.4f}, res_tpr={:.4f}, auc_score={:.4f}, tpr_h={:.4f}, tpr_m={:.4f}, tpr_l={:.4f}\n".format(
                        epoch + 1, test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l))
                log_file.flush()
                print("epoch:{:d}, test:  test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}, test_EER={:.4f}, res_tpr={:.4f}, auc_score={:.4f}, tpr_h={:.4f}, tpr_m={:.4f}, tpr_l={:.4f}\n".format(epoch + 1,
                                 test_APCER, test_BPCER, test_ACER, test_EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l))

            if test_ACER < best_test_ACER:
                best_test_ACER = test_ACER
                model_root_path = log + '/checkpoint/'
                if not os.path.exists(model_root_path):
                    os.makedirs(model_root_path)
                model_path = os.path.join(model_root_path, "fas_model_p{}_best.pth".format(args["pro"]))

                if args['muti_gpus'] == True:
                    if dist.get_rank() == 0:
                        torch.save({'state_dict': classifier.module.state_dict()}, model_path)
                        log_file.write("Fas model saved to {}\n".format(model_path))

                        print("[Best result] epoch:{}, test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}\n".format(epoch+1, test_APCER, test_BPCER, test_ACER))
                        log_file.write("[Best result] epoch:{}, test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}\n".format(epoch+1, test_APCER, test_BPCER, test_ACER))
                        log_file.flush()
                else:
                    torch.save({'state_dict': classifier.state_dict()}, model_path)
                    log_file.write("Fas model saved to {}\n".format(model_path))
                    print("[Best result] epoch:{}, test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}\n".format(epoch+1, test_APCER, test_BPCER, test_ACER))
                    log_file.write(
                        "[Best result] epoch:{}, test_APCER={:.4f}, test_BPCER={:.4f}, test_ACER={:.4f}\n".format(epoch+1, test_APCER, test_BPCER, test_ACER))
                    log_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--log", default='oppo', type=str)
    parser.add_argument('--pro', type=int, default=1, help='protocol: 1 or 2 or 3')
    parser.add_argument('--unknown', type=str, default='Replay', # Print Paper Replay
                        help="{'None','Co', 'Im', 'Ob','Half', 'Mann', 'Paper','Sil', 'Trans', 'Print','Eye', 'Funnyeye','Mouth', 'Paperglass','Replay'}")
    parser.add_argument('--clip', action='store_true', help="used clip_loss?")
    parser.add_argument('--adv_un', type=str, default='Facemask',
                        help="{'Facemask','Evolutionary'}")
    args = parser.parse_args()

    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Prepare the experiment folder
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    
    train(opts)

# CUDA_VISIBLE_DEVICES="2" python -m torch.distributed.launch --nproc_per_node 1 --master_port=12349 train_fas.py --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --use_fp16 True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --exp experiments/ffhq_34/ddpm.json --clip