'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from models import *
from utils import progress_bar

from ANSHB import *
import time

#%% Script Options
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 90
batch_size_train = 128
method = 'adam'


num_seeds = 4

#%% Load Data

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_train, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#%% Define and Run Networks

all_train_acc = np.zeros((num_seeds,num_epochs))
all_train_loss = np.zeros((num_seeds,num_epochs))

all_test_acc = np.zeros((num_seeds,num_epochs))
all_test_loss = np.zeros((num_seeds,num_epochs))

time_tracker = np.zeros((num_seeds,num_epochs))

best_acc = 0  # best test accuracy
def main():
    for rr in range(num_seeds):

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(43+rr)

        best_acc = 0  # best test accuracy

        # Model
        print('==> Building model..')
        # net = VGG('VGG19')
        net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss()

        if method == 'sgdm':
            wd = 5e-4
            lr = 0.1
            momentum = 0.9
            optimizer = optim.SGD(net.parameters(), lr=lr,
                                        momentum=momentum, weight_decay=wd)
        elif method =='asgd':
            wd =5e-4
            lr = 0.1
            optimizer = optim.ASGD(net.parameters(),lr=lr,weight_decay=wd)

        elif method == 'adam':
            wd = 5e-4
            lr = 0.001
            optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=wd)

        elif method == 'adamw':
            wd = 5e-4
            lr = 0.003
            optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=wd, amsgrad=False)

        elif method =='nadam':
            wd = 5e-4
            lr = 1e-3
            optimizer = optim.nadam(net.parameters(), lr=lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=wd, amsgrad=False)

        elif method =='adagrad':
            wd = 5e-4
            lr = 0.1
            optimizer = optim.Adagrad(net.parameters(),lr=lr, lr_decay=0, weight_decay=wd, initial_accumulator_value=0, eps=1e-10)

        elif method == 'anshb':
            wd = 5e-4
            lr = 0.4
            delta = 1e-2
            optimizer = NHBAdp(net.parameters(), lr=lr, weight_decay=wd, delta=delta)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1,last_epoch=-1)

        # Training
        def train(epoch):
            # print('\nEpoch: %d' % epoch)
            net.train()
            train_loss_tracker = []
            train_loss = 0
            correct = 0
            total = 0
            acc_tracker = []
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                optimizer.step()
            
                train_loss += loss.item()
                train_loss_tracker.append(loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                acc_tracker.append(100.*correct/total)
                
            acc = 100.*correct/total
            
                # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #               % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # print(train_loss_tracker)
            print('Epoch: %f | Train: Accuracy = %f | Loss = %f' %(epoch+1, np.mean(acc_tracker), np.mean(train_loss_tracker)))
            
            return np.mean(train_loss_tracker), acc, acc_tracker

        def test(epoch):
            global best_acc
            net.eval()
            test_loss_tracker = []
            acc_tracker = []
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    test_loss_tracker.append(loss.item())
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    acc_tracker.append(100.*correct/total)

                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            acc = 100.*correct/total
            
            if acc > best_acc:
                print(acc)
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                chckpt_name = method + '_checkpoint'
                if not os.path.isdir(chckpt_name):
                    os.mkdir(chckpt_name)
                torch.save(state, './'+chckpt_name+'/ckpt.pth')
                best_acc = acc
            
            return np.mean(test_loss_tracker), acc #np.mean(acc_tracker)

        test_loss_track = []
        test_acc_track = []
        train_loss_track = []
        train_acc_track = []

        for epoch in range(0, 90):
            start = time.time()
            train_loss_temp, train_acc_temp, acc_tracker = train(epoch)
            end = time.time()
            time_elapsed = end-start
            print('Time elapsed for last epoch: '+str(time_elapsed))
            time_tracker[rr,epoch-1] = time_elapsed
            test_loss_temp, test_acc_temp = test(epoch)
            scheduler.step()
            train_loss_track.append(train_loss_temp)
            train_acc_track.append(train_acc_temp)
            
            test_acc_track.append(test_acc_temp)
            test_loss_track.append(test_loss_temp)
            
            # if ((epoch+1) % 20) == 0:
            #     plt.plot(test_acc_track)
            #     plt.ylabel('Test Accuracy')
            #     plt.xlabel('Epochs')
            #     plt.grid()


        all_train_acc[rr,:] = train_acc_track 
        all_train_loss[rr,:] = train_loss_track 
        all_test_acc[rr,:] = test_acc_track 
        all_test_loss[rr,:] = test_loss_track 

        fname_acc_train = method + 'train_acc_rs'+str(rr)+'_cifar10.npy'
        fname_loss_train = method + 'train_loss_rs'+str(rr)+'_cifar10.npy'

        fname_acc = method + 'test_acc_rs'+str(rr)+'_cifar10.npy'
        fname_loss = method + 'test_loss_rs'+str(rr)+'_cifar10.npy'

        np.save(fname_acc_train, train_acc_track)
        np.save(fname_loss_train, train_loss_track)
        np.save(fname_acc, test_acc_track)
        np.save(fname_loss, test_loss_track)

if __name__=='__main__':
    main()
