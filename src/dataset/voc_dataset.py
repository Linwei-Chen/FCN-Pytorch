from torchvision import datasets
from torch.utils.data import DataLoader

import torch
import numpy as np
from dataset.segmentation_transform import Compose2, CenterCrop, RandomCrop, Normalize, RandomHorizontalFlip, ToTensor
from dataset.segmentation_transform import label_to_one_hot

CLASSES = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class.
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

VOC2012_RGB_mean = [x / 255.0 for x in [115.40426167733621, 110.06103779748445, 101.88298592833736]]
VOC2012_RGB_std = [x / 255.0 for x in [30.887643371028858, 31.592458778527075, 35.54421759971818]]


def image2label(img):
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(COLORMAP):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = np.array(img, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2])
    return np.array(cm2lbl[idx], dtype=np.int64)


def get_voc_data_loader(args, train=True):
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
    # year = args.dataset if year is None else year
    # print(f'*** year:{year}')

    voc_train_transform = Compose2([RandomCrop(size=args.crop_size, pad_if_needed=True),
                                    RandomHorizontalFlip(p=0.5),
                                    ToTensor(), Normalize(VOC2012_RGB_mean, VOC2012_RGB_std)])

    voc_val_transform = Compose2([CenterCrop(stride=args.stride),
                                  ToTensor(), Normalize(VOC2012_RGB_mean, VOC2012_RGB_std)])

    if train:
        voc_train_dataset = datasets.VOCSegmentation(root=args.voc_data_path,
                                                     year='2012',
                                                     image_set='train',
                                                     download=False,
                                                     transform=None,
                                                     target_transform=None,
                                                     transforms=voc_train_transform)
        return DataLoader(dataset=voc_train_dataset,
                          batch_size=args.batch_size,
                          shuffle=args.shuffle,
                          num_workers=args.prefetch,
                          pin_memory=True)
    else:
        voc_val_dataset = datasets.VOCSegmentation(root=args.voc_data_path,
                                                   year='2012',
                                                   image_set='val',
                                                   download=False,
                                                   transform=None,
                                                   target_transform=None,
                                                   transforms=voc_val_transform)
        return DataLoader(dataset=voc_val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=args.prefetch,
                          pin_memory=True)


def dataset_test():
    from train import config
    from model.fcn import FCN8s, VGGNet
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
        print(f'imgs:{imgs}\ntargets:{targets}')
        # targets = targets.copy_()
        # for i in targets.view(-1): print(i)
        outs = model(imgs)
        targets_one_hot = label_to_one_hot(targets, 21)
        # print(f'targets:{targets}\none-hot:{targets_one_hot}')
        targets_one_hot_argmax = targets_one_hot.argmax(dim=1, keepdim=True)
        print(f'targets_one_hot_argmax:{targets_one_hot_argmax}\ntargets:{targets}')
        print(f'check:{torch.eq(targets, targets_one_hot_argmax)}')
        # print(outs.type(), targets_one_hot.type())

        BCEWithLogitsLoss_check = nn.BCEWithLogitsLoss()(input=targets_one_hot * 1000, target=targets_one_hot)
        # BCEWithLogitsLoss_check = nn.BCELoss()(input=targets_one_hot, target=targets_one_hot)
        print(f'BCEWithLogitsLoss_check:{BCEWithLogitsLoss_check}')

        loss = criterion(outs, targets_one_hot)
        loss.backward()
        print(f'loss:{loss.item()}')
        loss_for_check = nn.CrossEntropyLoss()(targets_one_hot.reshape(1, 21, -1) * 10000, targets.reshape(1, -1))
        print(f'loss_for_check:{loss_for_check.item()}')
        zero_entropy = nn.CrossEntropyLoss()(torch.Tensor([[1000, 0, 0], [10000, 0, 0], [10000, 0, 0]]).float(),
                                             torch.Tensor([0, 0, 0]).long())
        print(f'zero_entropy:{zero_entropy}')
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
