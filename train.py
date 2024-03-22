""" the file contains utils to train a torch based model| built specifically for llama2"""
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainArgs
from model import Transformer


@torch.no_grad()
def estimate_loss(model: Transformer, eval_iters: int, train_dataloader: DataLoader, eval_dataloader: DataLoader) -> \
        Dict[
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

            logits, loss = model(inputs, 0, targets)
            losses[i] = loss.item()
        key = 'train' if dataloader == train_dataloader else 'eval'

        average_losses[key] = losses.mean().item()
    model.train()
    return average_losses


def rate(step: int, model_size: int, warmup: int, factor: int = 1):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def train(model: Transformer, train_config: TrainArgs, train_dataloader: DataLoader, eval_dataloader: DataLoader):
    optimizer = AdamW(model.parameters(), lr=train_config.lr)
    scheduler = LambdaLR(optimizer=optimizer,
                         lr_lambda=lambda step: rate(
                             step=step,
                             model_size=model.args.dim,
                             warmup=train_config.warmup_steps,
                         ))
    losses = []
    for epoch in tqdm(range(train_config.n_epochs)):
        model.train()
        for i, (X, Y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            logits, loss = model(X, 0, Y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log every log_interval batches
            if (i + 1) % train_config.log_interval == 0:
                out = estimate_loss(model=model,
                                    eval_iters=train_config.eval_iters,
                                    train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader)
                losses.extend(out)
                print(
                    f'Epoch: {epoch}, Batch: {i + 1}/{len(train_dataloader)} | train_loss: {out["train"]:.2f}, '
                    f'eval_loss: {out["eval"]:.2f}')

    return losses
