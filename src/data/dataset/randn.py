import os.path
import random
import re
import unicodedata
import torch
from torch.utils.data import Dataset
from PIL import Image

from typing import List, Union

def clean_filename(s):
    # 去除首尾空格和点号
    s = s.strip().strip('.')
    # 转换 Unicode 字符为 ASCII 形式
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
    illegal_chars = r'[/]'
    reserved_names = set()
    # 替换非法字符为下划线
    s = re.sub(illegal_chars, '_', s)
    # 合并连续的下划线
    s = re.sub(r'_{2,}', '_', s)
    # 转换为小写
    s = s.lower()
    # 检查是否为保留文件名
    if s.upper() in reserved_names:
        s = s + '_'
    # 限制文件名长度
    max_length = 200
    s = s[:max_length]
    if not s:
        return 'untitled'
    return s

def save_fn(image, metadata, root_path):
    image_path = os.path.join(root_path, str(metadata['filename'])+".png")
    Image.fromarray(image).save(image_path)

class RandomNDataset(Dataset):
    def __init__(self, latent_shape=(4, 64, 64), conditions:Union[int, List, str]=None, seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if isinstance(conditions, int):
            conditions = list(range(conditions)) # class labels
        elif isinstance(conditions, str):
            if os.path.exists(conditions):
                conditions = open(conditions, "r").read().splitlines()
            else:
                raise FileNotFoundError(conditions)
        elif isinstance(conditions, list):
            conditions = conditions
        self.conditions = conditions
        self.num_conditons = len(conditions)
        self.seeds = seeds

        if num_samples_per_instance > 0:
            max_num_instances = num_samples_per_instance*self.num_conditons
        else:
            max_num_instances = max_num_instances

        if seeds is not None:
            self.max_num_instances = len(seeds)*self.num_conditons
            self.num_seeds = len(seeds)
        else:
            self.num_seeds = (max_num_instances + self.num_conditons - 1)  // self.num_conditons
            self.max_num_instances = self.num_seeds*self.num_conditons
        self.latent_shape = latent_shape

    def __getitem__(self, idx):
        condition = self.conditions[idx//self.num_seeds]

        seed = random.randint(0, 1<<31) #idx % self.num_seeds
        if self.seeds is not None:
            seed = self.seeds[idx % self.num_seeds]

        filename = f"{clean_filename(str(condition))}_{seed}"
        generator = torch.Generator().manual_seed(seed)
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        metadata = dict(
            filename=filename,
            seed=seed,
            condition=condition,
            save_fn=save_fn,
        )
        return latent, condition, metadata
    def __len__(self):
        return self.max_num_instances

class ClassLabelRandomNDataset(RandomNDataset):
    def __init__(self, latent_shape=(4, 64, 64), num_classes=1000, conditions:Union[int, List, str]=None, seeds=None, max_num_instances=50000, num_samples_per_instance=-1):
        if conditions is None:
            conditions = list(range(num_classes))
        super().__init__(latent_shape, conditions, seeds, max_num_instances, num_samples_per_instance)
