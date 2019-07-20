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

CROP_SIZE =320

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
VOC2012_BGR_std = [x / 255.0 for x in [30.887643371028858, 31.592458778527075, 35.54421759971818]]


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
            img, target = t(img, target)
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

    def __call__(self, img, target):
        """

        :param input: tensor
        :return: if input is image return tensor[bs, 3, h, w],
        else if input is segment targets, return tensor[bs, 1, h, w]
        """
        # print(len(img.mode))
        # if input is img
        # if len(input.mode) == 3:

        # for img
        img = transforms.ToTensor()(img)
        # for target
        target = torch.from_numpy(np.array(target)).long()
        # print(target.type())
        zeros = torch.zeros(target.shape).long()
        # del 255.
        target = torch.where(target <= 20, target, zeros)
        target = target.unsqueeze(dim=0)
        return img, target


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)
            target = F.pad(target, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            target = F.pad(target, (int((1 + self.size[1] - target.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            target = F.pad(target, (0, int((1 + self.size[0] - target.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(target, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(target)
        # img.show(), target.show()
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_tensor, target):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(img_tensor, self.mean, self.std), target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


voc_transform = Compose2([RandomCrop(size=CROP_SIZE, pad_if_needed=True),
                                RandomHorizontalFlip(p=0.5),
                                to_tensor(), Normalize(VOC2012_RGB_mean, VOC2012_BGR_std)])


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
