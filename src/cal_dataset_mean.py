import os
import numpy as np
import cv2
from tqdm import tqdm

"""
VOC2012数据集的BGR平均值为
[101.88298592833736, 110.06103779748445, 115.40426167733621]

VOC2012数据集的BGR标准差为:
[35.54421759971818, 31.592458778527075, 30.887643371028858]

VOC图片尺寸
minh:71, minw:142, meanh:389.5076204379562, meanw:466.7975474452555

VOC2007数据集的BGR平均值为:
[99.95984164295817, 108.37466704041454, 114.37500234701753]

SBD 图片尺寸：
minh:88, minw:142, meanh:386.1916354556804, meanw:470.50613816063253

SBD 数据集的BGR平均值为:
[103.19907625613395, 111.6543631491467, 116.79657722131293]

SBD 数据集的BGR标准差为:
[35.49615257511964, 31.103248365270154, 30.670678993140825]

"""


def cal_imgs_mean(imgs_path):
    ims_list = os.listdir(imgs_path)
    B_means = []
    G_means = []
    R_means = []
    for im_list in tqdm(ims_list):
        try:
            # print(im_list)
            im = cv2.imread(os.path.join(imgs_path, im_list))
            # extrect value of diffient channel
            im_B = im[:, :, 0]
            im_G = im[:, :, 1]
            im_R = im[:, :, 2]
            # count mean for every channel
            im_B_mean = np.mean(im_B)
            im_G_mean = np.mean(im_G)
            im_R_mean = np.mean(im_R)
            # save single mean value to a set of means
            B_means.append(im_B_mean)
            G_means.append(im_G_mean)
            R_means.append(im_R_mean)
            # print('图片：{} 的 RGB平均值为 \n[{}，{}，{}]'.format(im_list, im_B_mean, im_G_mean, im_R_mean))
        except Exception:
            print(f'*** Loading {im_list} fail, jump to the next pic...')
    # three sets  into a large set
    a = [B_means, G_means, R_means]
    mean = [0, 0, 0]
    # count the sum of different channel means
    mean[0] = np.mean(a[0])
    mean[1] = np.mean(a[1])
    mean[2] = np.mean(a[2])
    print('数据集的BGR平均值为:\n[{}, {}, {}]'.format(mean[0], mean[1], mean[2]))


def cal_imgs_std(imgs_path, BGR_mean):
    """
    https://zh.wikipedia.org/zh/%E6%A8%99%E6%BA%96%E5%B7%AE
    标准差=方差的算术平方根=s=sqrt(((x1-x)^2 +(x2-x)^2 +......(xn-x)^2)/n)。
    \ SD= \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}
    {\displaystyle \mu } \mu 为平均值（ {\displaystyle {\overline {x}}} {\overline {x}}）。

    簡易口訣：離均差平方的平均；方均根。
    :param imgs_path:
    :param BGR_mean:
    :return:
    """
    ims_list = os.listdir(imgs_path)
    B = []
    G = []
    R = []
    for im_list in tqdm(ims_list):
        try:
            # print(im_list)
            im = cv2.imread(os.path.join(imgs_path, im_list))
            im_B = im[:, :, 0]
            im_G = im[:, :, 1]
            im_R = im[:, :, 2]
            # count mean for every channel
            im_B_mean = np.mean(im_B)
            im_G_mean = np.mean(im_G)
            im_R_mean = np.mean(im_R)
            B.append((im_B_mean - BGR_mean[0]) ** 2)
            G.append((im_G_mean - BGR_mean[1]) ** 2)
            R.append((im_R_mean - BGR_mean[2]) ** 2)
        except Exception:
            print(f'*** Loading {im_list} fail, jump to the next pic...')
    B_var = np.sqrt(np.mean(B))
    G_var = np.sqrt(np.mean(G))
    R_var = np.sqrt(np.mean(R))
    print('数据集的BGR标准差为:\n[{}, {}, {}]'.format(B_var, G_var, R_var))


def cal_imgs_size_mean(imgs_path):
    ims_list = os.listdir(imgs_path)
    minh, minw = float('inf'), float('inf')
    h_list, w_list = [], []
    for im_list in tqdm(ims_list):
        try:
            # print(im_list)
            im = cv2.imread(os.path.join(imgs_path, im_list))
            # h, w, c
            h, w, _ = im.shape
            h_list.append(h)
            w_list.append(w)
            minh, minw = min(minh, h), min(minw, w)
        except Exception:
            print(f'*** Loading {im_list} fail, jump to the next pic...')
    print(f'minh:{minh}, minw:{minw}, '
          f'meanh:{np.mean(h_list)}, meanw:{np.mean(w_list)}')


if __name__ == '__main__':
    # print('*** Cal the VOC2007')
    # cal_imgs_mean('/Users/chenlinwei/dataset/VOCdevkit/VOC2007/JPEGImages')

    voc2012 = '/Users/chenlinwei/dataset/VOCdevkit/VOC2012/JPEGImages'
    # cal_imgs_std(voc2012,
    #              [101.88298592833736, 110.06103779748445, 115.40426167733621])
    # cal_imgs_size_mean(voc2012)

    sbd = '/Users/chenlinwei/dataset/SBD/benchmark_RELEASE/dataset/img'
    # cal_imgs_size_mean(imgs_path=sbd)
    # cal_imgs_mean(imgs_path=sbd)
    cal_imgs_std(imgs_path=sbd, BGR_mean=[103.19907625613395, 111.6543631491467, 116.79657722131293])
