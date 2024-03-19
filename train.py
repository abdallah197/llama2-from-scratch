""" the file contains utils to train a torch based model| built specifically for llama2"""
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int, train_dataloader: DataLoader, eval_dataloader: DataLoader) -> Dict[
    str, float]:
    """
    Estimate the average loss of a model over a fixed number of iterations for both training and evaluation data.

    Parameters:
    - model (nn.Module): The torch model to evaluate.
    - eval_iters (int): The number of iterations to use for estimating the loss.
    - train_dataloader (DataLoader): DataLoader for the training data.
    - eval_dataloader (DataLoader): DataLoader for the evaluation data.

    Returns:
    - dict: A dictionary containing the average losses for the training and evaluation datasets.
    """
    model.eval()
    average_losses = {}

    for dataloader in [train_dataloader, eval_dataloader]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            inputs, targets = next(iter(dataloader))

            logits, loss = model(inputs, targets)
            losses[i] = loss.item()
        key = 'train' if dataloader == train_dataloader else 'eval'

        average_losses[key] = losses.mean().item()
    model.train()
    return average_losses
