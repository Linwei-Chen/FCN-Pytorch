import torch
from PIL import Image
from torchvision import transforms
import numpy as np

x = np.array(Image.open('/Users/chenlinwei/Downloads/VOC/VOC2012/SegmentationClass_aug/2010_002516.png'))
print(x[x.nonzero()])

sbd_train = open('/Users/chenlinwei/dataset/SBD_FULL11355/dataset/trainval.txt').readlines()
voc_2012_val = open('/Users/chenlinwei/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt').readlines()

voc_2012_val = [i.strip('\n') for i in voc_2012_val]
sbd_train = [i.strip('\n') for i in sbd_train]
print(voc_2012_val)
print(sbd_train)

print(f'{len(voc_2012_val)}, {len(sbd_train)}')
res = []
for i in sbd_train:
    if i not in voc_2012_val:
        res.append(i)

with open('train_ex_voc2012.txt', 'a+') as f:
    for i in res:
        f.write(i + '\n')

print(len(res))
