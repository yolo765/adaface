import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch.utils.data
from torch.nn import DataParallel
#from model.backbone import CBAMResNet
import torchvision.transforms as transforms
import argparse
import subprocess
import torch
import numpy as np
from tqdm import tqdm
import argparse
#import pandas as pd
from evaluation import tinyface_helper
# DataLoader
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import backbone.mobilefacenet_att
import glob, os
import random

random.seed(10)

class ListDataset(Dataset):
    def __init__(self, img_list, val_size):
        super(ListDataset, self).__init__()
        self.img_list = img_list
        
        self.val_size = val_size 
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Load Image
        image_path = self.img_list[idx]
        img = cv2.imread(image_path)
        try:
            img = img[:, :, :3]
        except:
            print('failed',image_path)

        if self.val_size != 112:
            img = cv2.resize(img, dsize=(self.val_size, self.val_size), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_NEAREST)


        # To Tensor
        #img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx



def prepare_dataloader(img_list, batch_size, num_workers=0, val_size=112):
    image_dataset = ListDataset(img_list, val_size)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader



def infer(model, dataloader, use_flip_test, has_norm):
    features = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            images = images.to("cuda")
            if not has_norm:
                feature = model(images)
            else:
                feature, norm = model(images)
            
            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                if not has_norm:
                    flipped_feature = model(fliped_images.to("cuda"))
                else:
                    flipped_feature, flipped_norm = model(fliped_images.to("cuda"))

                #fused_feature = (feature + flipped_feature) / 2
                fused_feature = np.concatenate((feature.data.cpu().numpy(), flipped_feature.data.cpu().numpy()), 1)
                #features.append(fused_feature.data.cpu().numpy())
                features.append(fused_feature)
            else:
                features.append(feature.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def load_model(args, checkpoint_path):
    # gpu init
    multi_gpus = False

    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda')

    net = backbone.mobilefacenet_att.build_model(model_name='mobilefacenet_att')

    # Load Pretrained Teacher
    net_teacher_ckpt = torch.load(checkpoint_path, map_location='cuda')['net_state_dict']
    net.load_state_dict(net_teacher_ckpt)


    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    net.eval()
    return net

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='my inference')
    parser.add_argument('--gpus', default='1', type=str)
    parser.add_argument('--batch_size', default=512, type=int, help='')
    parser.add_argument('--teacher_checkpoint_path', type=str, default='checkpoint/teacher/resol0-mbface_adaface/last_net.ckpt')
    parser.add_argument('--student_checkpoint_path', type=str, default='checkpoint/student/F_SKD/resol28-mbface_adaface-conv_features-outputbeforenorm/last_net.ckpt')
    args = parser.parse_args()
    
    # load model
    teacher_model = load_model(args,args.teacher_checkpoint_path)
    student_model = load_model(args,args.student_checkpoint_path)

    data_dir = '/home/teresa/data/data_aligned_3880/'
    id_list = os.listdir(data_dir)
    correct = 0
    total = 0
    val_res = (112,28)
    for i in range(100):
        gallery = '/home/teresa/data/data_aligned_3880/' + random.choice(id_list) + '/*'
        probe = '/home/teresa/data/data_aligned_3880/' + random.choice(id_list) + '/*'
        #gallery = '/home/teresa/data/data_aligned_3880_temp/1/*'
        #probe = '/home/teresa/data/data_aligned_3880_temp/2/*'
        assert(not probe==gallery)
        if not len(glob.glob(gallery)) == len(glob.glob(probe)):
            continue
        gallery_loader = prepare_dataloader(glob.glob(gallery), args.batch_size, num_workers=8, val_size=val_res[0])
        probe_loader = prepare_dataloader(glob.glob(probe), args.batch_size, num_workers=8, val_size=val_res[1])

        gallery_features = infer(teacher_model, gallery_loader, use_flip_test=False, has_norm=True)
        probe_features = infer(student_model, probe_loader, use_flip_test=False, has_norm=True)

        #gallery_features = gallery_features / np.linalg.norm(gallery_features, ord=2, axis=1).reshape(-1,1)
        #probe_features = probe_features / np.linalg.norm(probe_features, ord=2, axis=1).reshape(-1,1)
        #print(probe_features.shape)
        result = (probe_features @ gallery_features.T)
        #print(result)

        threshold = .43526/2

        print("negative", result.shape)
        greater = np.sum(result.flatten()>threshold)
        less =  np.sum(result.flatten()<=threshold)
        print('p', greater)
        print('n', less)
        correct = correct + less
        total = total + greater +less


        result = (probe_features @ probe_features.T)
        print("positive", result.shape)
        greater = np.sum(result.flatten()>threshold)
        less =  np.sum(result.flatten()<=threshold)
        print('p', greater)
        print('n', less)
        correct = correct + greater
        total = total + greater +less

        print('correct',correct,'total',total,'acc',correct/total)
