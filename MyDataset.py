import csv
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# 自定义Dataset类，读取CSV文件并加载图像
class HerbalMedicineDataset(Dataset):
    def __init__(self, csv_file, transform_big=None, transform_small=None):
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)
        self.transform_big = transform_big
        self.transform_small = transform_small
        self.nu = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像路径
        small_image_paths = self.data.iloc[idx, 0:4].tolist()  # 4张小图路径
        label = int(self.data.iloc[idx, 4])  # 标签，假设是数字
        if 'number' in self.data.columns:
            # print("文件中存在 'number' 这一列")
            number = int(self.data.iloc[idx, 5])
            self.nu = True
        small_images = [Image.open(img_path).convert('RGB') for img_path in small_image_paths]
        small_images = [img.resize((112, 112)) for img in small_images]  # 将小图调整为112x112
        small_image_concat = Image.new('RGB', (224, 224))  # 新建一个空白图像

        small_image_concat.paste(small_images[0], (0, 0))  # 左上 (0,0)
        small_image_concat.paste(small_images[1], (0, 112))  # 左下 (0,112)
        small_image_concat.paste(small_images[2], (112, 112))  # 右下 (112,112)
        small_image_concat.paste(small_images[3], (112, 0))  # 右上 (112,0)

        # 对拼接后的图像进行转换，拼接完成后再转换为Tensor
        if self.transform_small:
            small_image_concat = self.transform_small(small_image_concat)
        if self.nu:
            return small_image_concat, label, small_image_paths, number
        return small_image_concat, label, small_image_paths
