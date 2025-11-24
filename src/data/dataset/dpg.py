import torch
import json
import copy
import os
from torch.utils.data import Dataset
from PIL import Image

def dpg_save_fn(image, metadata, root_path):
    image_path = os.path.join(root_path, str(metadata['filename'])+"_"+str(metadata['seed'])+".png")
    Image.fromarray(image).save(image_path)

class DPGDataset(Dataset):
    def __init__(self, prompt_path, num_samples_per_instance, latent_shape):
        self.latent_shape = latent_shape
        self.prompt_path = prompt_path
        prompt_files = os.listdir(self.prompt_path)
        self.prompts = []
        self.filenames = []
        for prompt_file in prompt_files:
            with open(os.path.join(self.prompt_path, prompt_file)) as fp:
                self.prompts.append(fp.readline().strip())
                self.filenames.append(prompt_file.replace('.txt', ''))
        self.num_instances = len(self.prompts)
        self.num_samples_per_instance = num_samples_per_instance
        self.num_samples = self.num_instances * self.num_samples_per_instance

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance_idx = idx // self.num_samples_per_instance
        sample_idx = idx % self.num_samples_per_instance
        generator = torch.Generator().manual_seed(sample_idx)
        metadata = dict(
            prompt=self.prompts[instance_idx],
            filename=self.filenames[instance_idx],
            seed=sample_idx,
            save_fn=dpg_save_fn,
        )
        condition = metadata["prompt"]
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)
        return latent, condition, metadata