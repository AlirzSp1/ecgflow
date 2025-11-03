import os
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
import wfdb

from .data import data_factory

    
class EcgMimicIvDataset(Dataset):
    """MIMIC-IV ECG dataset
    """
    def __init__(self, data_instance, name, **kwargs):
        super().__init__()
        self.data = data_instance
        self.name = name
        debug = kwargs.get('debug', False)
        if not debug:
            self.id_list = data_instance.id_list
            self.record_list = data_instance.record_list
        else:
            self.id_list = data_instance.id_list[:1152]
            self.record_list = data_instance.record_list[:1152]
            
        self.trim_channels = data_instance.trim_channels
        self.transform = data_instance.transform
        
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        data = self.data
        rec = wfdb.rdrecord(self.record_list[idx])
        wf = rec.p_signal.T  # (12, time_samples)
        
        # swap (aVF,aVL)->(aVL,aVF) to match GE lead ordering
        wf = wf[(0,1,2,3,5,4,6,7,8,9,10,11),]
        if self.trim_channels:
            # take only I,II and precordial leads -> (8, time_samples)
            wf = wf[(0, 1) + tuple(range(6, 12)),]
        wf = torch.tensor(wf)
        if self.transform:
            for key in data.xkey_name:                
                wf = self.transform[key](wf)
        if wf.shape[0] == 1:
            wf = torch.squeeze(wf, axis=0)
        return wf.float(), 1


class EcgPtbxlDataset(Dataset):
    """ECG dataset for PTB-XL data.
    """
    def __init__(self, data_instance, name, **kwargs):
        super().__init__()
        self.data = data_instance
        self.name = name
        self.data_dir = data_instance.data_dir
        self.id_list = data_instance.id_list
        self.record_list = data_instance.record_list
        self.label_data = data_instance.label_data
        self.classes = data_instance.classes
        self.xkey_name = data_instance.xkey_name
        self.ykey = data_instance.ykey
        self.trim_channels = data_instance.trim_channels
        self.scale_y = data_instance.scale_y
        self.transform = data_instance.transform

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        xkey_name = self.xkey_name
        ykey = self.ykey
        label_data = self.label_data
        scale_y = self.scale_y
        transform = self.transform
        rec = wfdb.rdrecord(self.data_dir/self.record_list[idx])
        wf = rec.p_signal.T  # (12, 5000)
        if self.trim_channels:
            # take only I,II and precordial leads -> (8, 5000)
            wf = wf[(0, 1) + tuple(range(6, 12)),]
        wf = torch.tensor(wf)
        if transform:
            for key in xkey_name:
                wf = transform[key](wf)
        if wf.shape[0] == 1:
            wf = torch.squeeze(wf, axis=0)
        y = label_data[idx]
        if scale_y:
            if len(ykey) == 1:
                y = transform['y'](y)
            else:
                y = np.array([tx(y) for y,tx in zip(y, transform['y'])])
        return dict(X=wf.float(), label=torch.from_numpy(y).float())

    
def dataset_factory(name, **kwargs):
    """Return a Dataset instance after creating a Data instance with the
    given `name` and `kwargs` (cf.  .data.data_factory)
    """
    data_class = data_factory[name]
    _data = data_class(**kwargs)

    if name.startswith('mimic'):
        return EcgMimicIvDataset(_data, name, **kwargs)
    elif name.startswith('ptbxl'):
        return EcgPtbxlDataset(_data, name, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
