#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)
import torch.utils.data
from backbone.iresnet import iresnet50
from torch.nn import DataParallel
from margin.ArcMarginProduct import ArcMarginProduct
from margin.MultiMarginProduct import MultiMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.InnerProduct import InnerProduct
from utility.log import init_log
from utility.hook import feature_hook
from dataset.casia_webface import CASIAWebFace
from dataset.agedb import AgeDB30
from dataset.cfp import CFP_FP
from dataset.our3880 import Dataset3880
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import time
from evaluation.eval_lfw import evaluation_10_fold, getFeatureFromTorch, getFeatureFromTorchCrossResol
import numpy as np
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
import random

import backbone.mobilefacenet_att

def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def inference(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    if args.arch == 'iresnet50':
        checkpoint_dir_teacher = 'checkpoint/teacher_orig/resol0-IR/'
        #checkpoint_dir_student = checkpoint_dir_teacher
        checkpoint_dir_student = 'checkpoint/student_orig/F_SKD/resol28-IR'
        #checkpoint_dir_teacher = checkpoint_dir_student 
        checkpoint_dir_output = 'checkpoint/test_results/resol0-IR_resol28-IR'
    elif args.arch == 'mbface':
        #checkpoint_dir_teacher = 'checkpoint/teacher/resol0-mbface_cosface/'
        checkpoint_dir_teacher = 'checkpoint/teacher/resol0-mbface_adaface/'
        #checkpoint_dir_student = checkpoint_dir_teacher
        #checkpoint_dir_student = 'checkpoint/student/F_SKD/resol28-mbface_cosface/'
        #checkpoint_dir_student = 'checkpoint/student/F_SKD/resol28-mbface_cosface-conv_features-output' 
        checkpoint_dir_student = 'checkpoint/student/F_SKD/resol28-mbface_adaface-conv_features-outputbeforenorm'
        #checkpoint_dir_teacher = checkpoint_dir_student 
        checkpoint_dir_output = 'checkpoint/test_results/temp'
    
    os.makedirs(checkpoint_dir_output, exist_ok=True)


    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # define backbone and margin layer
    if args.arch == 'iresnet50':
        net_teacher = iresnet50(attention_type=args.mode)
        net_student = iresnet50(attention_type=args.mode)
    else:
        net_teacher = backbone.mobilefacenet_att.build_model(model_name='mobilefacenet_att')
        net_student = backbone.mobilefacenet_att.build_model(model_name='mobilefacenet_att')

    # Load Pretrained Teacher
    net_teacher_ckpt = torch.load(os.path.join(checkpoint_dir_teacher, 'last_net.ckpt'), map_location='cpu')['net_state_dict']
    net_teacher.load_state_dict(net_teacher_ckpt)
    # Load student
    net_student_ckpt = torch.load(os.path.join(checkpoint_dir_student, 'last_net.ckpt'), map_location='cpu')['net_state_dict']
    net_student.load_state_dict(net_student_ckpt)

    for param in net_teacher.parameters():
        param.requires_grad = False
    for param in net_student.parameters():
        param.requires_grad = False

    if multi_gpus:
        net_teacher = DataParallel(net_teacher).to(device)
        net_student = DataParallel(net_student).to(device)
    else:
        net_teacher = net_teacher.to(device)
        net_student = net_student.to(device)

    # test dataset
    net_teacher.eval()
    net_student.eval()
    print('Evaluation on 3880')

    os.makedirs(os.path.join(checkpoint_dir_output, 'result'), exist_ok=True)
    
    teacher_resol = 112
    student_resol = 28
    #print(args)

    eval_list = [(teacher_resol,student_resol)] # list of one tuple only
    for down_size in eval_list:
        data_dir = '/home/teresa/data/data_aligned_3880/'
        our3880dataset = Dataset3880(data_dir, 'data_aligned_3880_temp.txt', down_size, transform=transform)
        our38880loader = torch.utils.data.DataLoader(our3880dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

        # test model on AgeDB30
        print('testing teacher: ', checkpoint_dir_teacher)
        print('testing student: ', checkpoint_dir_student)        
        getFeatureFromTorchCrossResol(os.path.join(checkpoint_dir_output, 'result/cur_3880_result.mat'), net_teacher,net_student, device, our3880dataset, our38880loader, True)
        age_accs = evaluation_10_fold(os.path.join(checkpoint_dir_output, 'result/cur_3880_result.mat'))
        try:
            print('Evaluation Result on 3880 %dX - %.2f' %(down_size, np.mean(age_accs) * 100))
        except:
            print('Evaluation Result on 3880 %d,%dX - %.2f' %(down_size[0],down_size[1], np.mean(age_accs) * 100))

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='/home/teresa/projects/teaching-where-to-look/Face/')
    #parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    #parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/teacher/iresnet50-ir/last_net.ckpt', help='model save dir')
    parser.add_argument('--mode', type=str, default='ir', help='attention type', choices=['ir', 'cbam'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--arch', type=str, default='mbface') #iresnet50
    args = parser.parse_args()


    # Path
    args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
    args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')
    args.lfw_test_root = os.path.join(args.data_dir, 'evaluation/lfw')
    args.lfw_file_list = os.path.join(args.data_dir, 'evaluation/lfw.txt')
    args.agedb_test_root = os.path.join(args.data_dir, 'evaluation/agedb_30')
    args.agedb_file_list = os.path.join(args.data_dir, 'evaluation/agedb_30.txt')
    args.cfpfp_test_root = os.path.join(args.data_dir, 'evaluation/cfp_fp')
    args.cfpfp_file_list = os.path.join(args.data_dir, 'evaluation/cfp_fp.txt')


    # Seed
    set_random_seed(args.seed)
    
    # Run    
    inference(args)
