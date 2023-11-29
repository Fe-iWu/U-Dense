#coding=utf-8
#coding=gbk
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,5"


import argparse

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--train_txt',default="../../../nas2/path/LT/tougue_cancer/3D_data/train_seg_2.txt",help="train paths")
parser.add_argument('--valid_txt',default="../../../nas2/path/LT/tougue_cancer/3D_data/valid_seg_2.txt",help="valid paths")
parser.add_argument('--test_txt',default="../../../nas2/path/LT/tougue_cancer/3D_data/test_seg_2.txt",help="test paths")
parser.add_argument('--model_path', default="../../../nas2/path/LT/tougue_cancer/3D_data/Densenet_lossseg/model_save8",help="save model path")
parser.add_argument('--fig_path', default="../../../nas2/path/LT/tougue_cancer/3D_data/Densenet_lossseg", help="result fig path")
parser.add_argument('--result_path', default="../../../nas2/path/LT/tougue_cancer/3D_data", help="result path")
parser.add_argument('--lr',type=float, default=0.0001, help="learning_rate")
parser.add_argument('--batchsize',type=int, default=4, help="input batch size")
parser.add_argument('--epochs',type=int, default=100, help="the number of epochs to train for")
parser.add_argument('--num_classes',type=int, default=2, help="the number of classes")
parser.add_argument('--num_workers',type=int, default=4,help="num workers")

args = parser.parse_args(args=[])


device_ids = [0,1]

class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class AllData(Dataset):
    def __init__(self, path_data):
        self.img_paths = []
        self.img_labels = []
        self.rows = 150
        self.cols = 150
        with open(path_data, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.img_paths.append(line)
                if 'pos' in line:
                    self.img_labels.append(1)
                else:
                    self.img_labels.append(0)
        self.img_labels = torch.Tensor(self.img_labels)
        # F.one_hot(self.img_labels.to(torch.int64), num_classes=2)

    def __getitem__(self, idx):
        datapath = self.img_paths[idx]
        label = self.img_labels[idx]
        img = np.load(datapath)
        img = img/255.0
        frame = img.shape[0]
        img = torch.Tensor(np.array(img))
        pad_dims = (0, 0, 0, 0, 0, 64 - frame)
        img_pad = F.pad(img, pad_dims, "constant")
        return img_pad.unsqueeze(0), label

    def __len__(self):
        return len(self.img_paths)


class _DenseLayer_3d(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock_3d(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer_3d(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)

class _Transition_3d(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet_3d(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0.5,
                 num_classes=1000):

        super().__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, num_init_features,
                                kernel_size=(3, 7, 7),
                                stride=(2, 2, 2),
                                padding=(3, 3, 3),
                                bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))]))

        # Each denseblock
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock_3d(num_layers=num_layers,
                                   num_input_features=num_features,
                                   bn_size=bn_size,
                                   growth_rate=growth_rate,
                                   drop_rate=drop_rate)

            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition_3d(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def save_img(epochs,train,valid,label):
    plt.plot(range(epochs),train,range(epochs),valid)
    plt.xlabel("epoch")
    plt.ylabel(label)
    plt.legend(["train "+label, "valid "+label], loc="upper right")
    save_fig_path = args.fig_path+'/'+label+'.jpg'
    plt.savefig(save_fig_path)
    plt.close()


train_data = AllData(args.train_txt)
valid_data = AllData(args.valid_txt)
test_data = AllData(args.test_txt)

train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
valid_dataloader = DataLoader(valid_data,batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)
test_dataloader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

model = DenseNet_3d(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=args.num_classes)
#device = torch.device("cuda:4")
model = model.cuda(device_ids[0])
model = nn.DataParallel(model,device_ids=device_ids)
#model = model.to(device)

# loss_fn = nn.CrossEntropyLoss()
loss_fn = FocalLoss(class_num=args.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
save_model_num = 0
train_acc = []
train_loss = []
valid_acc = []
valid_loss = []

min_acc_num = 13
acc_threshold = 75

for epoch in range(args.epochs):
    print("------第{}轮训练ʼ".format(epoch + 1))
    model.train()
    train_sum_loss = 0
    train_accurate_num = 0
    for data in train_dataloader:
        dicoms, labels = data
        dicoms = dicoms.cuda()
        labels = labels.cuda()
        outputs = model(dicoms)
        loss = loss_fn(outputs, labels.long())
        train_sum_loss += loss
        train_accurate_num += (outputs.argmax(1) == labels).sum()
        optimizer.zero_grad()  # �Ż�ǰ�ݶ�����
        loss.backward()
        optimizer.step()
        # print("Loss: {}".format(loss.item()))
    average_train_loss = train_sum_loss / len(train_dataloader)
    train_loss.append(average_train_loss)
    train_accuracy = (train_accurate_num / (len(train_dataloader)*args.batchsize))*100
    train_acc.append(train_accuracy)
    print('训练集准确率:{}%'.format(train_accuracy.cpu().numpy()))

    valid_sum_loss = 0
    valid_accurate_num = 0
    model.eval()
    with torch.no_grad():
        for data in valid_dataloader:
            dicoms, labels = data
            dicoms = dicoms.cuda()
            labels = labels.cuda()
            outputs = model(dicoms)
            loss = loss_fn(outputs, labels.long())
            valid_sum_loss = valid_sum_loss + loss
            valid_accurate_num += (outputs.argmax(1) == labels).sum()
            print(" output and label are following")
            print(outputs.tolist())
            print(labels.tolist())
    print('accurate_num: {}         total_num: {}'.format(valid_accurate_num, len(valid_dataloader)*args.batchsize))
    print('验证集准确率:{:.2f}%'.format(valid_accurate_num / len(valid_dataloader) * 25))

    average_valid_loss = valid_sum_loss / len(valid_dataloader)
    valid_loss.append(average_valid_loss)
    valid_accuracy = (valid_accurate_num / (len(valid_dataloader)*args.batchsize))*100
    valid_acc.append(valid_accuracy)

    meanloss = np.mean(torch.tensor(valid_loss).cpu().numpy())
    if epoch == 99:
        save_model_num+=1
        save_path = args.model_path+'/model-'+str(save_model_num)+'.pth'
        print("save model_______________")
        torch.save(model.state_dict(), save_path)
    if valid_accuracy > acc_threshold and average_valid_loss < meanloss or valid_accuracy > 80:
        # min_acc_num = valid_accurate_num
        save_model_num+=1
        save_path = args.model_path+'/model-'+str(save_model_num)+'.pth'
        print("save model_______________")
        torch.save(model.state_dict(), save_path)
    

save_img(args.epochs, torch.tensor(train_loss).cpu().numpy(), torch.tensor(valid_loss).cpu().numpy(), "loss")
save_img(args.epochs, torch.tensor(train_acc).cpu().numpy(), torch.tensor(valid_acc).cpu().numpy(), "accuracy")

modelfile = os.listdir(args.model_path)
model_path = args.model_path + '/' + modelfile[-1]
#model = DenseNet_3d(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=args.num_classes)

#model = model.cuda(device_ids[0])
#model = nn.DataParallel(model,device_ids=device_ids)
model.load_state_dict(torch.load(model_path))

accurate_num = 0

for data in test_dataloader:
    dicoms, labels = data
    dicoms = dicoms.cuda()
    labels = labels.cuda()
    outputs = model(dicoms)
    accurate_num += (outputs.argmax(1) == labels).sum()
accuracy = accurate_num / (len(test_dataloader)*args.batchsize)
print("test accuracy: {:.2f}".format(accuracy))
