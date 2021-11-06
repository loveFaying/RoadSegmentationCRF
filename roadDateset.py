import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2


class RoadDataset(Dataset):

    def __init__(self, phase='train', root='/home/houyw/experiment/dataset_512/'):
        self.phase = phase  #
        self.root = root  # 数据集的根目录

    def __len__(self):
        return len(os.listdir(os.path.join(self.root+self.phase, 'data')))

    def __getitem__(self, index):
        file_name_data = os.listdir(os.path.join(self.root+self.phase, 'data'))
        file_name_data = sorted(file_name_data)[index]
        file_name_label = os.listdir(os.path.join(self.root+self.phase, 'label'))
        file_name_label = sorted(file_name_label)[index]

        img = cv2.imread(os.path.join(self.root+self.phase, 'data', file_name_data))
        mask = cv2.imread(os.path.join(self.root+self.phase, 'label', file_name_label), cv2.IMREAD_GRAYSCALE)

        mask = np.expand_dims(mask, axis=2)

        # 归一化
        # 讲图像从[0,255]规范化到[-1.6,1.6]
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img, mask, file_name_data, file_name_label

def get_loader(phase, batch_size=1):
    dataset = RoadDataset(phase=phase)
    if phase == 'test':     # 测试阶段为了方便对比，不进行shuffle
        loader = DataLoader(
            dataset,
            batch_size=batch_size
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
    return loader


if __name__ == '__main__':
    train_loader = get_loader(phase='train', batch_size=4)
    valid_loader = get_loader(phase='valid', batch_size=4)
    test_loader = get_loader(phase='test')

    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))

    for index, (img, label, file_name_data, file_name_label) in enumerate(train_loader):
        print(index)
        print(img.shape)
        print(label.shape)
        print(file_name_data)
        print(file_name_label)

        length = label.shape[0]
        for i in range(length):
            print(label[i,:,:,:].shape)
            print(file_name_data[i])
            print(file_name_label[i])







