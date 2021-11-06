from __future__ import generator_stop
import os
import argparse

from PIL import Image
import torch.nn.functional as F
import torchvision.utils as vutils
import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm
from icecream import ic

from models.UNet import get_unet_model

from roadDateset import get_loader
from evaluate import mDC_batch, mIOU_batch
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
test_loader = get_loader(phase='test')
model_name = "UNet"
save_path = "./results/"+model_name
if not os.path.isdir(save_path):
    os.makedirs(save_path)
model_path = "weights/UNet_mass.th"  # AF_DUNet path

# model architecture

if model_name == 'UNet' or model_name == 'UNet_CRF':
    model = get_unet_model()

model = model.cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

IOU = 0.
DC = 0.
length = len(test_loader)

for index, (img, _, file_name_data, file_name_label) in tqdm(enumerate(test_loader)):
    img = img.to(device)

    mask = model.forward(img)
    mask = mask.cpu().data.numpy()

    from postprocessing.postprocessing import Postprocessing
    from postprocessing.pp_config_p1 import Config
    img_GT_PIL = Image.open(os.path.join('/home/houyw/experiment/dataset_512', 'test/data/') + str(file_name_data[0]), "r")
    img_GT_PIL = np.array(img_GT_PIL).astype(np.uint8)
    con = Config()
    post = Postprocessing(con)
    MAP = post.crf(image=img_GT_PIL, prediction=mask.squeeze())
    MAP = torch.tensor(MAP).unsqueeze(0).unsqueeze(0)

    DC += mDC_batch(pred=MAP, filename=file_name_label, root=os.path.join('/home/houyw/experiment/dataset_512', 'test/label/'))
    IOU += mIOU_batch(pred=MAP, filename=file_name_label, root=os.path.join('/home/houyw/experiment/dataset_512', 'test/label/'))

mDC = DC / length
mIOU = IOU / length
print('mIOU: {}'.format(mIOU))
print('mDC: {}'.format(mDC))



for index, (img, _, file_name_data, file_name_label) in tqdm(enumerate(test_loader)):
    img = img.to(device)
    mask = model.forward(img)
    mask = mask.squeeze().cpu().data.numpy()

    # ic(os.path.join('../dataset_512', 'test/data/') + str(file_name[0]))
    img_GT = cv2.imread(os.path.join('/home/houyw/experiment/dataset_512', 'test/data/') + str(file_name_data[0]))
    label_GT = cv2.imread(os.path.join('/home/houyw/experiment/dataset_512', 'test/label/') + str(file_name_label[0]), cv2.IMREAD_GRAYSCALE)

    ## TODO: mask_UNet
    mask_UNet = mask
    mask_UNet[mask_UNet > 0.5] = 255
    mask_UNet[mask_UNet <= 0.5] = 0

    img_GT_PIL = Image.open(os.path.join('/home/houyw/experiment/dataset_512', 'test/data/') + str(file_name_data[0]), "r")
    img_GT_PIL = np.array(img_GT_PIL).astype(np.uint8)
    ## TODO: mask_UNet_CRF


    from postprocessing.postprocessing import Postprocessing
    from postprocessing.pp_config_p9 import Config
    con = Config()
    post = Postprocessing(con)
    MAP = post.crf(image=img_GT_PIL, prediction=mask)

    # plt.subplot(2, 2, 1)
    # plt.imshow(img_GT)
    # plt.title("img_GT")
    # plt.subplot(2, 2, 2)
    # plt.imshow(label_GT)
    # plt.title("label_GT")
    # plt.subplot(2, 2, 3)
    # plt.imshow(mask)
    # plt.title("UNet")
    # plt.subplot(2, 2, 4)
    # plt.imshow(MAP)
    # plt.title("UNet_CRF")
    # plt.show()
    
    cv2.imwrite(os.path.join(save_path, str(index) + '_pred.png'), mask_UNet.astype(np.uint8))
    cv2.imwrite(os.path.join(save_path, str(index) + '_img.png'), img_GT)
    cv2.imwrite(os.path.join(save_path, str(index) + '_label.png'), label_GT)
