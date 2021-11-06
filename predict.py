from __future__ import generator_stop
import os
import argparse

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

device = torch.device('cuda:0')
test_loader = get_loader(phase='test')
model_name = "UNet"
save_path = "./results/"+model_name
if not os.path.isdir(save_path):
    os.makedirs(save_path)
model_path = "weights/"+model_name+".th"  # AF_DUNet path

# model architecture

if model_name == 'UNet':
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

    DC += mDC_batch(pred=mask, filename=file_name_label, root=os.path.join('/home/houyw/experiment/dataset_512', 'test/label/'))
    IOU += mIOU_batch(pred=mask, filename=file_name_label, root=os.path.join('/home/houyw/experiment/dataset_512', 'test/label/'))

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

    
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    
    cv2.imwrite(os.path.join(save_path, str(index) + '_pred.png'), mask.astype(np.uint8))
    cv2.imwrite(os.path.join(save_path, str(index) + '_img.png'), img_GT)
    cv2.imwrite(os.path.join(save_path, str(index) + '_label.png'), label_GT)
