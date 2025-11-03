"""Tools and setup to evaluate models
"""

from typing import Type, Union, Tuple, Dict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
from timm import utils
from tqdm import tqdm


def model_predict(
    model: Type[nn.Module], # torch model instance
    data_loader: Type[torch.utils.data.DataLoader],  # data loader instance
    device: Literal['cpu', 'cuda'] = 'cuda',  # device to use for inference
    return_eval: bool = True,  # return dict of evaluation metrics?
    return_roc: bool = False,  # return ROC curves?
    regression: bool = False,  # is this a regression task?
    num_classes: int = 2,  # number of output classes
    debug: bool = False  # return y_true?
    ) -> Union[torch.Tensor,
               Dict,
               Tuple[Dict, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor],
               Tuple[Dict, torch.Tensor, torch.Tensor, 
                     Type[utils.MetricCollection]]
               ]:
    if return_eval:
        if not regression:
            if num_classes == 2:
                metrics = utils.get_binary_torchmetrics(
                    compute_with_cache=False)
            elif num_classes > 2:
                metrics = utils.get_multilabel_torchmetrics(
                    num_classes, compute_with_cache=False)
    model.eval()
    with torch.no_grad():
        y_pred, y_true = [], []
        #for batch_idx, (samples, y) in enumerate(data_loader):
        for batch_d in tqdm(data_loader):
            samples, y = batch_d['X'], batch_d['label']
            samples = samples.to(device)
            output = model(samples)
            if regression:
                y_pred.append(output)
            elif num_classes == 2:
                y_pred.append(output.softmax(dim=1))
            elif num_classes > 2:
                y_pred.append(output.sigmoid())
            else:
                y_pred.append(output)
            if return_eval:
                y_true.append(y)
                if not regression:
                    if num_classes == 2:
                        metrics.update(output[:,1].cpu(), y[:,0].cpu().int())
                    elif num_classes > 2:
                        metrics.update(output.cpu(), y.cpu().int())
                    metrics_d = metrics.compute()
        y_pred = torch.cat(y_pred).cpu().numpy()
        if return_eval:
            y_true = torch.cat(y_true).numpy()
            if not regression:
                roc = metrics_d.pop('ROC', None)
                metrics_d = {k:v.item() for k,v in metrics_d.items()}
    if return_eval:
        if debug:
            if not regression:
                return metrics_d, y_pred, y_true, metrics
            else:
                return y_pred, y_true
        elif return_roc:
            return metrics_d, roc
        else:
            return metrics_d
    else:
        return y_pred
    