"""
The Recorder class enables visualizing attention weights.

Code from:
https://github.com/lucidrains/vit-pytorch?tab=readme-ov-file#accessing-attention

"""

import torch
from torch import nn

from timm.models.vision_transformer import Attention


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Recorder(nn.Module):
    def __init__(self, vit, device = None):
        super().__init__()
        self.vit = vit
        self.transformer = dict()
        if hasattr(vit, 'blocks'):
            self.transformer = dict(blocks=vit.blocks)
            self.recordings = dict(blocks=[])
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _blocks_hook(self, _, input, output):
        self.recordings['blocks'].append(output.clone().detach())

    def _register_hook(self):
        for key,txf in self.transformer.items():
            if key == 'block':
                hook = self._blocks_hook
            modules = find_modules(txf, Attention)
            for module in modules:
                handle = module.attend.register_forward_hook(hook)
                self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        for recordings in self.recordings.values():
            recordings.clear()

    def record(self, attn, key='blocks'):
        recording = attn.clone().detach()
        self.recordings[key].append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        attns = dict()
        for key,recordings in self.recordings.items():
            recordings = tuple(map(lambda t: t.to(target_device), recordings))
            attns[key] = torch.stack(recordings, dim = 1) if len(recordings) > 0 else None
            
        return pred, attns
