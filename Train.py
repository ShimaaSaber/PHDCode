# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:52:26 2021

@author: Shimaa
"""

# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

import time, os, yaml
from Model.Network import Network


from random_erasing import RandomErasing
from LoadFolder import LoadFolder


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

version = torch.__version__
print(' version   ',version)
torch.cuda.empty_cache()



######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
#parser.add_argument('--gpu_ids', default='0,1,2', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_net', type=str, help='output model name')
parser.add_argument('--data_dir', default='Dataset/market1501', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batchsize', default=28, type=int, help='batchsize')
parser.add_argument('--lr', default=0.04, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--model', default=Network)
opt = parser.parse_args()


print(' options ', opt)
  
use_gpu = torch.cuda.is_available()
if  use_gpu:
    # use_gpu = False
    print(' GPU ', use_gpu)

data_dir = opt.data_dir
name = opt.name

######################################################################
# Load Data
# ---------
#
print('start load train data')
transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
print('start load validation data')
transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]


data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = LoadFolder(os.path.join(data_dir, 'train_all'),data_transforms['train'])
image_datasets['val'] = LoadFolder(os.path.join(data_dir, 'val'),data_transforms['val'])

batch = {}

class_names = image_datasets['train'].classes
class_vector = [s[1] for s in image_datasets['train'].samples]
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=0)   #num_workers=8
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


since = time.time()



y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    
    for epoch in range(num_epochs):
       
        # Each epoch has a training and validation phase
        for phase in ['train']:
            model.train(True)  # Set model to training mode


            running_loss = 0.0
            running_corrects = 0.0
            
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape

                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # print('zero the parameter gradients')
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                               
                
                _, preds = torch.max(outputs.data, 1)
                loss_id = criterion(outputs, labels)
                

                loss = loss_id 

                # backward + optimize only if in training phase

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item()  # * opt.batchsize
                    
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0]
                    
                running_corrects += float(torch.sum(preds == labels.data))
                print(' {} train {} '.format(i ,  len(dataloaders[phase])))
                
            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / datasize
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            
            
            last_model_wts = model.state_dict()

        print()


    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    path_name = './Weights'
    save_path = os.path.join(path_name, name, save_filename)
    
    
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
# Finetuning the convnet


model = opt.model(len(class_names))

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

ignored_params =  list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40, 60, 80], gamma=0.1)

######################################################################
# Train and evaluate
dir_name = os.path.join('./Weights', name)

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

if __name__ == '__main__':
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)