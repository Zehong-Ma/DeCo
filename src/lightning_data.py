from typing import Any
import torch
import time
import copy
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, IterableDataset
from src.data.dataset.randn import RandomNDataset

def mirco_batch_collate_fn(batch):
    batch = copy.deepcopy(batch)
    new_batch = []
    for micro_batch in batch:
        new_batch.extend(micro_batch)
    x, y, metadata = list(zip(*new_batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata

def collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except:
            pass
    x = torch.stack(x, dim=0)
    return x, y, stacked_metadata

def eval_collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    x = torch.stack(x, dim=0)
    return x, y, metadata

class DataModule(pl.LightningDataModule):
    def __init__(self,
                train_dataset:Dataset=None,
                eval_dataset:Dataset=None,
                pred_dataset:Dataset=None,
                train_batch_size=64,
                train_num_workers=16,
                train_prefetch_factor=8,
                eval_batch_size=32,
                eval_num_workers=4,
                pred_batch_size=32,
                pred_num_workers=4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pred_dataset = pred_dataset
        # stupid data_convert override, just to make nebular happy
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor


        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size

        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers

        self._train_dataloader = None

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        micro_batch_size = getattr(self.train_dataset, "micro_batch_size", None)
        if micro_batch_size is not None:
            assert self.train_batch_size % micro_batch_size == 0
            dataloader_batch_size = self.train_batch_size // micro_batch_size
            train_collate_fn = mirco_batch_collate_fn
        else:
            dataloader_batch_size = self.train_batch_size
            train_collate_fn = collate_fn
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size

        # build dataloader sampler
        if not isinstance(self.train_dataset, IterableDataset):
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=global_rank)
        else:
            sampler = None

        self._train_dataloader = DataLoader(
            self.train_dataset,
            dataloader_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            collate_fn=train_collate_fn,
            sampler=sampler,
        )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.eval_dataset, self.eval_batch_size,
                          num_workers=self.eval_num_workers,
                          prefetch_factor=2,
                          sampler=sampler,
                          collate_fn=eval_collate_fn
                )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.pred_dataset, batch_size=self.pred_batch_size,
                          num_workers=self.pred_num_workers,
                          prefetch_factor=4,
                          sampler=sampler,
                          collate_fn=eval_collate_fn
               )
