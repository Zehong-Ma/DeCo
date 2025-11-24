import torch
import json
import copy
from torch.utils.data import Dataset
import os
from PIL import Image

def geneval_save_fn(image, metadata, root_path):
    path = os.path.join(root_path, metadata['filename'])
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # save image
    image_path = os.path.join(path, "samples", f"{metadata['seed']}.png")
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    Image.fromarray(image).save(image_path)
    # metadata_path
    metadata_path = os.path.join(path, "metadata.jsonl")
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp)

class GenEvalDataset(Dataset):
    def __init__(self, meta_json_path, num_samples_per_instance, latent_shape):
        self.latent_shape = latent_shape
        self.meta_json_path = meta_json_path
        with open(meta_json_path) as fp:
            self.metadatas = [json.loads(line) for line in fp]
        self.num_instances = len(self.metadatas)
        self.num_samples_per_instance = num_samples_per_instance
        self.num_samples = self.num_instances * self.num_samples_per_instance

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance_idx = idx // self.num_samples_per_instance
        sample_idx = idx % self.num_samples_per_instance
        metadata = copy.deepcopy(self.metadatas[instance_idx])
        generator = torch.Generator().manual_seed(sample_idx)
        condition = metadata["prompt"]
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)
        filename = f"{idx}"
        metadata["seed"] = sample_idx
        metadata["filename"] = filename
        metadata["save_fn"] = geneval_save_fn
        return latent, condition, metadata