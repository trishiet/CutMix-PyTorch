# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

from __future__ import print_function
import argparse
import os
import os.path
import glob
import sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np
import cv2
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from numpy import asarray
from numpy import moveaxis 
from torchvision.datasets import STL10
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--label_blend', default='cutmix', type=str,
                    help='type of label blending')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100


"""
class myDataset(Dataset):
    def __init__(self, train):
        self.cifar100 = datasets.CIFAR100(root='../data', download=True, transform = None)

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        data, target = self.cifar100[idx]

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        data = transform_train(data)

        target_3d = np.zeros((32, 32), dtype = np.int64)
        target_3d.fill(int(target))
        target = target_3d

        return data, target, idx
"""

class STL10Alpha(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(idx) + '_sal_fuse.png')
        image = Image.open(img_name)
        image = image.resize((224, 224))
        image = np.asarray(image)
        """
        image = io.imread(img_name)
        print(image)
        print(np.shape(image))
        """
        
        return image

class myDataSet(STL10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                myDataset('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                myDataset('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'stl10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train=transforms.Compose([
            transforms.RandomCrop(96),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])
        train_loader = torch.utils.data.DataLoader(
            myDataSet('../data', split='train', download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            myDataSet('../data', split='test', transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 10

    elif args.dataset == 'imagenet':
        traindir = os.path.join('/home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    elif args.dataset == 'caltech256':
        cal_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        calset = Caltech256(root='./data', download=True, transform=cal_transform)

        train_loader = torch.utils.data.DataLoader(
            calset(train = True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            calset(train = False),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 257

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # Learning rate parameter
    current_LR = get_learning_rate(optimizer)[0]
    # print("current_LR", current_LR)

    stl10_alpha = STL10Alpha(root_dir = 'alpha_images')

    # i is index
    # input is list BATCH_SIZE image matrices
    # target is list of BATCH_SIZE label (ints)
    # index is list of BATCH_SIZE image ids
    for i, (input, target, index) in enumerate(train_loader):
        # print("i, (input, target, index)", i, (input, target, index))
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)

        # beta parameter, 1.0
        # print("args.beta", args.beta)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            # lam is random decimal between 0 and 1
            lam = np.random.beta(args.beta, args.beta)
            #print("lam", lam)

            # take vector of [0, 1, ..., BATCH_SIZE - 1] and shuffle these indices
            rand_index = torch.randperm(input.size()[0])
            #print("rand_index", rand_index)

            # the original batch labels vector
            target_a = target
            #print("target_a", target_a)

            # For each index in rand_index , target_b[i] = target_a[index]
            target_b = target[rand_index]
            #print("target_b", target_b)

            # top left and bottom right points of bounding box
            # y starts at 0 at the top and increases going down
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            #print("bounding box (x1, y1, x2, y2)", bbx1, bby1, bbx2, bby2)

            # Normalizing and making sure alpha image is of correct dims and bounding box is cut
            alpha_image = []
            for j in index:
                one_plane = []
                for k in range(3):
                    one_plane.append(stl10_alpha[j])
                alpha_image.append(one_plane)

            alpha_image = np.array(alpha_image, dtype=np.float32)
            alpha_mask = np.clip(alpha_image[:, :, bbx1:bbx2, bby1:bby2], 0, 1)
            #alpha_mask_channel_first = moveaxis(alpha_mask, 2, 0)
            '''
            print(alpha_mask)
            print(alpha_mask.shape)
            foreground_alpha_tensor = torch.from_numpy(alpha_mask)
            print(foreground_alpha_tensor.size())
            print(type(foreground_alpha_tensor.data))

            background_alpha_tensor = torch.from_numpy(1 - alpha_mask)
            print(background_alpha_tensor.size())
            print(type(background_alpha_tensor.data))

            existing = input.cpu()[:, :, bbx1:bbx2, bby1:bby2]
            print(existing.size())
            print(type(existing.data))

            foreground = torch.mm(input.cpu()[:, :, bbx1:bbx2, bby1:bby2], foreground_alpha_tensor)
            background = torch.mm(input.cpu()[rand_index, :, bbx1:bbx2, bby1:bby2].numpy(), background_alpha_tensor)
            combined = torch.add(foreground, background)
            print(combined.shape)
            input[:, :, bbx1:bbx2, bby1:bby2] = combined
            '''

            foreground_input = input.cpu()[:, :, bbx1:bbx2, bby1:bby2].numpy()
            foreground = np.multiply(foreground_input, alpha_mask)
            background = np.multiply(input.cpu()[rand_index, :, bbx1:bbx2, bby1:bby2].numpy(), 1-alpha_mask)
            input[:, :, bbx1:bbx2, bby1:bby2] = torch.from_numpy(np.add(foreground, background))

            # lambda is 1 - the ratio of bounding box area to total image area
            # lambda is also the weight applied to the loss for unshuffled batch images
            # 1 - lambda is the weight applied to the loss portion of shuffled batch images

            if args.label_blend == 'cutmix':
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            elif args.label_blend == 'cutblend':
                ratio = np.count_nonzero(alpha_mask) + 0.000001 / (np.size(alpha_mask) + 0.000001)
                lam = 1 - (ratio * (bbx2 - bbx1) * (bby2 - bby1)) / (input.size()[-1] * input.size()[-2])
            elif args.label_blend == 'alphas-only':
                lam = 1 - (np.count_nonzero(alpha_mask) + 0.000001) / (np.count_nonzero(alpha_image) + 0.000001)

            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, index) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet') or args.dataset == 'stl10':
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
