import numpy as np
import torch
import cv2
import torch.nn as nn

# Pixel Accuracy
# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
# 阶级失衡的问题   不使用mAP 作为评价指标
SMOOTH = 1e-6
# Intersection-Over-Union

def filter_model(f):
#   if f.endswith('gen_checkpoint.th'):
    if f.endswith('.th'):
        return True
    else:
        return False

def iou_coef(pred, target):

    pred = np.array(pred.squeeze())
    target = np.array(target.squeeze())

    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = (np.sum(intersection)+SMOOTH) / (np.sum(union)+SMOOTH)
    return iou_score


def dice_coef(pred, target, empty_score=1.0):
    """Calculates the dice coefficient for the images"""
    # print(im1.shape)
    # print(im2.shape)
    pred = np.asarray(pred).astype(np.bool)
    target = np.asarray(target).astype(np.bool)

    if pred.shape != target.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = pred > 0.5
    im2 = target > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum

def mIOU_batch(pred, filename, root, fine_tune=False):
    length = pred.shape[0]
    IOU = 0.
    for i in range(length):
        image = pred[i, :, :, :]
        image = np.squeeze(image)
        if fine_tune:
            image[image>0] = 255
            image[image<=0] = 0
        else:
            image[image>0.5] = 255
            image[image<=0.5] = 0

        label_GT = cv2.imread(root + str(filename[i]), cv2.IMREAD_GRAYSCALE)
        IOU += iou_coef(pred=image, target=label_GT)
    return IOU / length

def mDC_batch(pred, filename, root, fine_tune=False):
    length = pred.shape[0]
    DC = 0.
    for i in range(length):
        image = pred[i, :, :, :]
        image = np.squeeze(image)
        if fine_tune:
            image[image>0] = 255
            image[image<=0] = 0
        else:
            image[image>0.5] = 255
            image[image<=0.5] = 0

        label_GT = cv2.imread(root + str(filename[i]), cv2.IMREAD_GRAYSCALE)
        DC += dice_coef(pred=image, target=label_GT)
    return DC / length

if __name__ == '__main__':
    # device = torch.device('cuda')  # cuda:0
    # inputs = torch.rand(3, 512, 512).unsqueeze(0).to(device)
    # print(inputs.shape)
    #
    # net = get_fcns_model().to(device)
    # res = net(inputs)  # res是一个tuple类型
    # print('res shape:', res.shape)
    X = torch.tensor([
        [10, 1],
        [0.1, 0]
    ])
    Y = torch.tensor([
        [2, 10],
        [1, 1]
    ])
    print(iou_coef(X, Y))
    print(dice_coef(X, Y))
