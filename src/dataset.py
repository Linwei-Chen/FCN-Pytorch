import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from os import path as osp
import numbers
import random
import torch
import numpy as np
import torchvision.transforms.functional as F

CROP_SIZE = 128

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class.
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def image2label(img):
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(COLORMAP):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = np.array(img, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2])
    return np.array(cm2lbl[idx], dtype=np.int64)


class Compose2(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img = t(img)
            target = t(target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class to_tensor(object):
    def __init__(self):
        pass

    def __call__(self, input):
        """

        :param input: tensor
        :return: if input is image return tensor[bs, 3, h, w],
        else if input is segment targets, return tensor[bs, 1, h, w]
        """
        # print(len(img.mode))
        # if input is img
        if len(input.mode) == 3:
            return transforms.ToTensor()(input)
        # if img is target
        else:
            target = torch.from_numpy(np.array(input)).long()
            # print(target.type())
            zeros = torch.zeros(target.shape).long()
            # del 255
            target = torch.where(target <= 20, target, zeros)
            target = target.unsqueeze(dim=0)
            return target


voc_transform = Compose2([transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomCrop(size=CROP_SIZE, pad_if_needed=True),
                          to_tensor()])


def get_voc_data_loader(args, train=True):
    voc_dataset = datasets.VOCSegmentation(root=args.data_path,
                                           year=args.dataset,
                                           image_set='train' if train else 'val',
                                           download=False,
                                           transform=None,
                                           target_transform=None,
                                           transforms=voc_transform)
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
    return DataLoader(dataset=voc_dataset,
                      batch_size=args.batch_size,
                      shuffle=args.shuffle,
                      num_workers=args.prefetch,
                      pin_memory=True)


def label_to_one_hot(targets, n_class):
    """

    :param targets: long tensor[bs, 1, h, w]
    :param nlabels: int
    :return: float tensor [bs, nlabel, h, w]
    """
    # batch_size, _, h, w = targets.size()
    # res = torch.zeros([batch_size, nlabels, h, w])
    targets = targets.squeeze(dim=1)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=n_class)
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


def dataset_test():
    from train import config
    from fcn import FCN8s, VGGNet
    import torch.nn as nn
    import torch.optim as Opim

    args = config()
    model = FCN8s(pretrained_net=VGGNet(requires_grad=True), n_class=21)
    optimizer = Opim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    dataset = get_voc_data_loader(args)
    for _, (imgs, targets) in enumerate(dataset):
        optimizer.zero_grad()
        print(imgs.size(), targets.size())
        # targets = targets.copy_()
        # for i in targets.view(-1): print(i)
        outs = model(imgs)
        targets_one_hot = label_to_one_hot(targets, 21)
        # print(f'targets:{targets}\none-hot:{targets_one_hot}')
        targets_one_hot_argmax = targets_one_hot.argmax(dim=1, keepdim=True)
        print(f'targets_one_hot_argmax:{targets_one_hot_argmax}\ntargets:{targets}')
        print(f'check:{torch.eq(targets, targets_one_hot_argmax)}')
        # print(outs.type(), targets_one_hot.type())
        loss = criterion(outs, targets_one_hot)
        loss.backward()
        print(f'loss:{loss.item()}')
        loss_for_check = nn.CrossEntropyLoss()(targets_one_hot.reshape(1, 21, -1), targets.reshape(1, -1))
        print(loss_for_check.item())
        optimizer.step()
        pass


def cross_entropy_test():
    y = torch.tensor([10, 9]).long()
    x = torch.nn.functional.one_hot(y, 11).float()
    print(f'x:{x}\ny:{y}')
    print(f'x:{x.size()}\ny:{y.size()}')
    loss = torch.nn.CrossEntropyLoss()(input=x, target=y)
    print(f'loss:{loss.item()}')
    pass


if __name__ == '__main__':
    # cross_entropy_test()
    dataset_test()
    pass
