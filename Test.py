# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:35:54 2021

@author: Shimaa Saber
"""

from __future__ import print_function, division

import argparse
import torch
from torchvision import datasets, transforms
import os
import scipy.io

from Model.Network import Network

import json
import time

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')  # last
parser.add_argument('--test_dir', default='Dataset/market1501', type=str, help='./test_data')
parser.add_argument('--name', default='ft_net', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')

opt = parser.parse_args()


network_name = Network


print('     options ', opt)


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def main():
    ###load config###
    print('load the training config')
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    print('path model name', config_path)
    
    name = opt.name
    test_dir = opt.test_dir

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    print('loading data  ')
    data_dir = test_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=0) for x in
                       ['gallery', 'query']}
    use_gpu = torch.cuda.is_available()
    print(' ---- GPU--- ', use_gpu)
    ######################################################################
    # Load model
    # ---------------------------
    print('loading model')

    def load_network(network):
        save_path = os.path.join('./Weights', name, 'net_%s.pth' % opt.which_epoch)
        network.load_state_dict(torch.load(save_path))
        return network

    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(model, dataloaders):

        features = torch.FloatTensor()
        count = 0
        count_each = 0
        for data in dataloaders:
            start_time = (time.time())
            img, label = data
            n, c, h, w = img.size()
            count += n
            
            ff = torch.FloatTensor(n, 751).zero_().to(device)
            # ff = torch.FloatTensor(n, 1000).zero_().to(device)

            
            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = img.to(device)
                
                outputs = model(input_img)
                ff += outputs
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            ff = ff.data.cpu().float()
            features = torch.cat((features, ff), 0)
            elapsed = (time.time() - start_time)
            print(' Extract feature    Epoch:[ {} / {} ]    ****  Time  (h:m:s): {}.'
                  .format(count_each,len(dataloaders),elapsed))
            count_each += 1
        return features

    def get_id(img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            # filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    
    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    model_structure = network_name(751)
    model = load_network(model_structure)

    # Remove the final fc layer and classifier layer

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.to(device)

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'])
        query_feature = extract_feature(model, dataloaders['query'])
        
    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}

    
    scipy.io.savemat('net_test.mat', result)
    
if __name__ == '__main__':
    main()
