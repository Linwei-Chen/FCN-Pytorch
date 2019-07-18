# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import sys
import os
import argparse
from logger import Logger, ModelSaver
from fcn import get_model
from dataset import get_voc_data_loader, label_to_one_hot
from tqdm import tqdm

def config():
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--data_path', type=str, default='/Users/chenlinwei/dataset',
                        help='Root for the voc dataset.')
    parser.add_argument('--dataset', type=str, default='2012',
                        choices=['2007', '2012'], help='Choose between voc2007/2012.')
    # Optimization options
    parser.add_argument('--optimizer_name', '-op', type=str, default='Adam', help='Optimizer to train model.')
    parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=2e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--schedule', type=int, nargs='+', default=[250, 400],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--save_steps', '-ss', type=int, default=100, help='steps to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

    # Architecture
    parser.add_argument('--model_name', type=str, default='fcn8s', help='choose from fcn32s，16s, 8s, s.')
    parser.add_argument('--n_class', type=int, default=21, help='classes number to classify.')

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=12, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')
    args = parser.parse_args()
    args.optimizer_name = args.optimizer_name.lower()
    if args.save is None:
        args.save = f'../{args.model_name}_{args.dataset}'
    if args.log is None:
        args.log = f'../{args.model_name}_{args.dataset}'

    args.scheduler_name = f'{args.optimizer}_scheduler'
    print(args)
    return args


# n_class = 20
# batch_size = 6
# epochs = 500
# lr = 1e-4
# momentum = 0
# w_decay = 1e-5
# step_size = 50
# gamma = 0.5
# configs = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(
#     batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
# print("Configs:", configs)


def get_device(args):
    return torch.device('gpu' if args.ngpu >= 1 and torch.cuda.is_available() else 'cpu')


def model_accelerate(args, model):
    if args.ngpu > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0 and torch.cuda.is_available():
        model.cuda()
    return model


def get_optimizer(args, name: str, model):
    op_dict = {
        'adam': optim.Adam(model.parameters(), lr=args.lr),
        'sgd': optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    }
    assert isinstance(name, str)
    assert name.lower() in [op.lower() for op in op_dict.keys()]
    return op_dict.get(name)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# train function (forward, backward, update)
def train_one_epoch(args, model, optimizer, train_loader, logger, model_saver):
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    device = get_device(args)
    for step, (imgs, targets) in enumerate(train_loader, start=1):
        t1 = time.perf_counter()
        optimizer.zero_grad()
        targets_one_hot = label_to_one_hot(targets, n_class=args.n_class)
        imgs, targets_one_hot = imgs.to(device), targets_one_hot.to(device)
        outs = model(imgs)
        loss = criterion(input=outs, target=targets_one_hot)
        loss.backward()
        optimizer.step()
        t2 = time.perf_counter()
        print(f'step:{step} | loss:{loss.item():.8f} | lr:{get_lr(optimizer)} | time:{t2 - t1}')
        logger.log(key='train_loss', data=loss.item())
        if step % args.save_steps == 0:
            logger.visualize(key='train_loss', range=(-1000, -1))
            logger.save_log()
            model_saver.save(name=args.model_name, model=model)
            model_saver.save(name=args.optimizer_name, model=optimizer)
            # break


def val(args, model, scheduler, val_loader, logger, model_saver):
    model.eval()
    device = get_device(args)
    total_ious = []
    pixel_accs = []
    for step, (imgs, targets) in enumerate(val_loader):
        print(f'***processing : {step}/{len(val_loader)}')
        imgs = imgs.to(device)
        # targets_one_hot = label_to_one_hot(targets, n_class=args.n_class)
        # targets_one_hot = targets_one_hot.to(device)

        outs = model(imgs).data.cpu().numpy()

        N, _, h, w = outs.shape
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, args.n_class).argmax(axis=1).reshape(N, h, w)
        target = targets.cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            temp_iou = iou(args, p, t)
            temp_pa = pixel_acc(p, t)
            print(f'iou:{temp_iou} pixel_accuracy:{temp_pa}')
            total_ious.append(temp_iou)
            pixel_accs.append(temp_pa)
        # break

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    miou = np.nanmean(ious)
    pixel_accs = np.array(pixel_accs).mean()
    epoch_now = scheduler.state_dict()['last_epoch']
    print("*** epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch_now, pixel_accs, miou, ious))
    if logger.get_max(key='meanIoU') < miou or logger.get_max(key='meanPixel') < pixel_accs:
        model_saver.save(name=args.model_name + f'_mIU:{ious}_mp:{pixel_accs}', model=model)
    logger.log(key='meanIoU', data=miou, show=True)
    logger.log(key='IoUs', data=ious, show=True)
    logger.log(key='meanPixel', data=pixel_accs, show=True)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(args, pred, target):
    ious = []
    for cls in range(args.n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    # init the tools
    args = config()
    logger = Logger(save_path=args.save, json_name=args.optimizer_name)
    model_saver = ModelSaver(save_path=args.save, name_list=[args.optimizer_name, args.model_name])

    # get model
    model = get_model(name=args.model_name, n_class=args.n_class)
    model_saver.load(name=args.model_name, model=model)

    # get optimizer
    optimizer = get_optimizer(args, name=args.optimizer_name, model=model)
    model_saver.load(name=args.optimizer_name, model=optimizer)

    # Accelerate the model training
    device = get_device(args)
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=args.gamma)
    model_saver.load(name=args.scheduler_name , model=scheduler)

    # Main loop
    for _ in range(args.epochs):
        # get dataset
        train_loader = get_voc_data_loader(args, train=True)
        val_loader = get_voc_data_loader(args, train=False)

        # val(args, model, scheduler, val_loader, logger, model_saver)

        scheduler.step()
        epoch_now = scheduler.state_dict()['last_epoch']
        print(f'*** Epoch now:{epoch_now}.')

        if epoch_now >= args.epochs:
            print('*** Training finished!')
        train_one_epoch(args, model, optimizer, train_loader, logger, model_saver)
        val(args, model, scheduler, val_loader, logger, model_saver)

        model_saver.save(name=args.scheduler_name , model=scheduler)
        logger.save_log()
        logger.visualize()