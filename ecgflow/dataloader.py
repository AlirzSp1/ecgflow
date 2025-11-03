"""Data loader for ECGs.

Simplifies timm.create_loader() mainly for eval use.
"""

from pathlib import Path
import random
from typing import Callable, Tuple
from functools import partial

import numpy as np
from timm.data import create_dataset, create_loader
import torch

from .transforms import create_transforms

MODEL_BEST = 'model_best.pth.tar'
DEVICE = 'cuda'
DATASET_NAME = 'ecgflow/ptbxl_diag'
SPLIT = 'test'
BATCH_SIZE = 128
INPUT_SIZE = (8, 10000)


def get_dataset(data_dir, dataset_name=DATASET_NAME, split=SPLIT,
                batch_size=BATCH_SIZE, input_size=INPUT_SIZE,
                **kwargs):
    return create_dataset(
        name=dataset_name, root=data_dir, split=split, 
        batch_size=batch_size, input_size=input_size, **kwargs)

 
def get_dataloader(data_dir, dataset_name=DATASET_NAME, split=SPLIT,
                   batch_size=BATCH_SIZE, input_size=INPUT_SIZE,
                   is_training=(SPLIT=='train'), num_workers=10,
                   shuffle=False, **kwargs):
    dataset = get_dataset(
        data_dir, dataset_name=dataset_name, split=split, 
        batch_size=batch_size, input_size=input_size, **kwargs)
    data_loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=batch_size,
        is_training=is_training,
        no_aug=True,
        num_workers=num_workers,
        device=DEVICE,
        use_prefetcher=False,
        pin_memory=False,
        ecgflow_input=True,
        shuffle=shuffle
    )
    return data_loader


def _worker_init(worker_id, worker_seeding='all'):
    # Borrowed from timm data/loader.py
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter 
        # class w/ worker_info.seed.
        # to reproduce some old results (same seed + hparam combo), 
        # partial seeding is required (skip numpy re-seed).
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def create_eval_loader(
    dataset: torch.utils.data.Dataset,
    input_size: Tuple[int, int], 
    batch_size: int = 128, 
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    worker_seeding: str = 'all',
    pin_memory: bool = False,
    is_training: bool = False,
    ):
    dataset.transform = create_transforms(dataset, input_size)
    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=None,
        collate_fn=torch.utils.data.default_collate,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    return torch.utils.data.DataLoader(dataset, **loader_args)
