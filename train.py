""" the file contains utils to train a torch based model| built specifically for llama2"""
import logging
from typing import Dict

import deepspeed
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainArgs
from model import Transformer


@torch.no_grad()
def estimate_loss(model: Transformer, eval_iters: int, train_dataloader: DataLoader, eval_dataloader: DataLoader,
                  device: str) -> Dict[str, float]:
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
            inputs, targets = inputs.to(device), targets.to(device)

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


def train(model: Transformer, train_config: TrainArgs, train_dataloader: DataLoader, eval_dataloader: DataLoader,
          args: Dict):
    optimizer = AdamW(model.parameters(), lr=train_config.lr)
    scheduler = LambdaLR(optimizer=optimizer,
                         lr_lambda=lambda step: rate(
                             step=step,
                             model_size=model.args.dim,
                             warmup=train_config.warmup_steps,
                         ))
    if args.deepspeed:
        deepspeed.init_distributed()
        logging.info('Deepspeed is enabled.')
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=scheduler,
            dist_init_required=False
        )
    losses = []
    best_eval_loss = float('inf')

    for epoch in tqdm(range(train_config.n_epochs)):
        model.train()
        for i, (X, Y) in enumerate(train_dataloader):
            logits, loss = model(X, 0, Y)
            if args.deepspeed:
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Log every log_interval batches
            if (i + 1) % args['log_interval'] == 0:
                out = estimate_loss(model=model,
                                    eval_iters=args['eval_iters'],
                                    train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader,
                                    device=args['device'])
                losses.extend([out])
                print(
                    f'Epoch: {epoch}, Batch: {i + 1}/{len(train_dataloader)} | train_loss: {out["train"]:.2f}, '
                    f'eval_loss: {out["eval"]:.2f}')

            # save the model if it was outperforming the previous best model
            cur_eval_loss = losses[-1]['eval']
            if cur_eval_loss < best_eval_loss:
                torch.save(model.state_dict(), f"best_model_eval{cur_eval_loss:.2nf}_epoch{epoch}.pth")
                cur_eval_loss = best_eval_loss
                print(f"New best model saved with eval_loss: {cur_eval_loss:.2f}")

    return losses
