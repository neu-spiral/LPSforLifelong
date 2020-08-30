import random
import pickle
import math
import os
import sys
import numpy as np
import torch
import time
np.set_printoptions(threshold=sys.maxsize)

from testers import *
from utils import *
import DataGenerator as DG
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class CVTrainValTest():
    def __init__(self, base_path,save_path):

        self.base_path = base_path
        self.save_path = save_path
        print(base_path)
        print(save_path)

    def load_data_cifar(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))
        
        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

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
        
        self.training_set = DG.CifarDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.CifarDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return self.train_generator

    def load_data_mnist(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))
        
        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

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
        
        self.training_set = DG.MnistDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MnistDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return self.train_generator

    def load_data_mixture(self, params):
        '''
        Mixture dataset contains 5 tasks, [mnist,cifar,mnist,cifar,mnist]
        Mnist > Cifar => subsample mnist
        # Mnist: 60000
        # Cifar: 5000
        '''
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))
        
        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

        # scale number of training sample
        scale = 1
        trigger = False
        if len(self.y_train) > 5000:
            trigger=True
            params.epochs = 50
            params.epochs_prune = 30
            params.epochs_mask_retrain = 50
            print('Sample {} examples in each training epoch.'.format(int(len(self.y_train)*scale)))
        else:
            params.epochs = 300
            params.epochs_prune = 200
            params.epochs_mask_retrain = 300
            
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
        
        self.training_set = DG.MixtureDataGenerator(self.x_train, self.y_train, scale=scale, trigger=trigger)
        DataParams = {'batch_size': params.batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MixtureDataGenerator(self.x_test, self.y_test, trigger=trigger)
        DataParams = {'batch_size': params.batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return params, self.train_generator

    
    def train_model(self, args, model, masks, train_loader, criterion, optimizer, scheduler, epoch):
        atch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        idx_loss_dict = {}
        
        #if masks:
        #    test_sparsity_mask(args,masks)
        model.train()
        
        for i, (input, target) in enumerate(train_loader):
            input = input.float().cuda()
            target = target.long().cuda()
            scheduler.step()
            # compute output
            output = model(input)
            ce_loss = criterion(output, target)

            # measure accuracy and record loss
            prec1,_ = accuracy(output, target, topk=(1,5))
            losses.update(ce_loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            ce_loss.backward()
            
            if masks:
                with torch.no_grad():
                    for name, W in (model.named_parameters()):
                        # fixed-layers are shared layers for multi-tasks, it should not be trained besides the first task
                        if name in args.fixed_layer:
                            W.grad *= 0
                            continue
                        if name in masks and name in args.pruned_layer:
                            W.grad *= 1-masks[name].cuda()
            
            optimizer.step()

            # print(i)
            if i % 100 == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                print('({0}) lr:[{1:.5f}]  '
                      'Epoch: [{2}][{3}/{4}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                      .format('adam', current_lr,
                       epoch, i, len(train_loader), loss=losses, top1=top1))
            if i % 100 == 0:
                idx_loss_dict[i] = losses.avg
        return model
    
    def test_model(self, args, model, mask=""):
        
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        top1 = AverageMeter()

        if mask:
            set_model_mask(model, mask)
        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.test_generator):
            input = input.float().cuda()
            target = target.long().cuda()
            
            # compute output
            output = model(input)
            output = output.float()
            
            # measure accuracy and record loss
            prec1,_ = accuracy(output, target, topk=(1,5))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Testing Prec@1 {top1.avg:.3f}%'.format(top1=top1))

        return top1.avg
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
