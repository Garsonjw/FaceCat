'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pdb

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res





def get_threshold(score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        #pdb.set_trace()
        type = float(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type==1:
            num_real += 1
        else:
            num_fake += 1
    print("num_real")
    print(num_real)
    print("num_fake")
    print(num_fake)
    print("count")
    print(count)
    min_error = count    # account ACER (or ACC)
    min_threshold = 0.0
    min_ACC = 0.0
    min_ACER = 0.0
    min_APCER = 0.0
    min_BPCER = 0.0
    
    
    for d in data:
        threshold = d['map_score']
        
        type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
        type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])
        
        ACC = 1-(type1 + type2) / count
        APCER = type2 / num_fake
        BPCER = type1 / num_real
        ACER = (APCER + BPCER) / 2.0
        
        if ACER < min_error:
            min_error = ACER
            min_threshold = threshold
            min_ACC = ACC
            min_ACER = ACER
            min_APCER = APCER
            min_BPCER = min_BPCER

    # print(min_error, min_threshold)
    return min_threshold, min_ACC, min_APCER, min_BPCER, min_ACER



def test_threshold_based(threshold, score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type==1:
            num_real += 1
        else:
            num_fake += 1
    
    type1 = 0
    type2 = 0
    for s in data:
        if s['map_score'] <= threshold and s['label'] == 1:
            type1+=1
        elif s['map_score'] > threshold and s['label'] == 0:
            type2 +=1
    
    ACC = 1-(type1 + type2) / count
    APCER = type2 / num_fake
    BPCER = type1 / num_real
    ACER = (APCER + BPCER) / 2.0
    
    return ACC, APCER, BPCER, ACER

def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
  
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th

def performance_val(score_val_filename):

    # val 
    with open(score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  #label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0

def performances(score_val_filename, score_test_filename):

    # val 
    with open(score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  #label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0
    
    
    
    # test 
    with open(score_test_filename, 'r') as file2:
        lines = file2.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold 
    type1 = 0
    type2 = 0
    success = ""

    for s in data:
        if s['map_score'] <= val_threshold and s['label'] == 1:
            type1+=1
            success = success+"B"
        elif s['map_score'] > val_threshold and s['label'] == 0:
            type2+=1
            success = success+"A"
        else:
            success = success+"S"  
    
    with open(score_test_filename, 'r+') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            lines[i] =f"{line} {success[i]}\n"
        f.seek(0)
        f.writelines(lines)
        f.truncate()

    test_ACC = 1-(type1 + type2) / count
    test_APCER = type2 / num_fake
    if num_real !=0:
        test_BPCER = type1 / num_real
    else:
        test_BPCER = 0
    test_ACER = (test_APCER + test_BPCER) / 2.0
    
    return val_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER

def get_tpr_at_fpr(tpr_lst, fpr_lst, score_lst, fpr_value):
	"""returns true postive rate and threshold given false positive rate value."""
	abs_fpr = np.absolute(fpr_lst - fpr_value)
	idx_min = np.argmin(abs_fpr)
	fpr_value_target = fpr_lst[idx_min]
	idx = np.max(np.where(fpr_lst == fpr_value_target))
	return tpr_lst[idx], score_lst[idx]

def my_metrics(score_test_filename):
    # val
    with open(score_test_filename, 'r') as file:
        lines = file.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    success = ""
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, thresholds = roc_curve(test_labels, test_scores) #false positive rate, true positive rate, thresholds
    auc_score = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    tnr = 1 - fpr
    EER0 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    EER1 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = min(EER0, EER1)
    best_ACER, best_AP, best_BP = 100, 100, 100
    best_threshold = 100

    for idx_ in range(len(tpr)):
        _tpr, _fpr = tpr[idx_], fpr[idx_]
        _tnr, _fnr = tnr[idx_], fnr[idx_]
        assert _tpr + _fnr == 1, print(_tpr, _fnr)
        assert _tnr + _fpr == 1, print(_tnr, _fpr)
        APCER = _fpr / (_fpr + _tnr)
        BPCER = _fnr / (_fnr + _tpr)
        ACER = 0.5 * (APCER + BPCER)
        if ACER < best_ACER:
            best_ACER = ACER
            best_AP = APCER
            best_BP = BPCER
            best_threshold = thresholds[idx_]

    ## fnr == 0.5% as the first PAMI paper version.
    abs_fnr = np.absolute(fnr - 0.005)
    idx = np.argmin(abs_fnr)
    res_tpr = tpr[idx]

    tpr_h, _ = get_tpr_at_fpr(tpr, fpr, thresholds, 0.005)
    tpr_m, _ = get_tpr_at_fpr(tpr, fpr, thresholds, 0.01)
    tpr_l, _ = get_tpr_at_fpr(tpr, fpr, thresholds, 0.02)


    for s in data:
        if s['map_score'] <= best_threshold and s['label'] == 1:
            success = success + "B"
        elif s['map_score'] > best_threshold and s['label'] == 0:
            success = success + "A"
        else:
            success = success + "S"
    with open(score_test_filename, 'r+') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            lines[i] = f"{line} {success[i]}\n"
        f.seek(0)
        f.writelines(lines)
        f.truncate()

    return best_AP, best_BP, best_ACER, EER, res_tpr, auc_score, tpr_h, tpr_m, tpr_l






def performances_SiW_EER(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0
    
    
    
    return val_threshold, val_ACC, val_APCER, val_BPCER, val_ACER







def performances_SiWM_EER(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0
    
    
    
    return val_threshold, val_err, val_ACC, val_APCER, val_BPCER, val_ACER




def get_err_threhold_CASIA_Replay(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
  
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th, right_index



def performances_CASIA_Replay(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    
    FRR = 1- tpr    # FRR = 1 - TPR
    
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate
    
    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index]





def performances_ZeroShot(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    
    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    
    FRR = 1- tpr    # FRR = 1 - TPR
    
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate
    
    return val_ACC, auc_val, HTER[right_index]






def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
