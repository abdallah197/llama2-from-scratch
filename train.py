""" the file contains utils to train a torch based model| built specifically for llama2"""
import logging
from typing import Dict

import deepspeed
import pandas as pd
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


def save_ds_checkpoint(step, epoch, model, ckpt_id, args, optimizer=None, lr_scheduler=None):
    """Save a model checkpoint."""
    if args['deepspeed']:
        client_state = {'step': step, 'epoch': epoch}
        saved_path = model.save_checkpoint(args['save_dir'], ckpt_id, client_state=client_state)
        if saved_path is None:
            logging.info('Failed to save deepspeed checkpoint.')
        else:
            logging.info(f'saved checkpoint to {saved_path}')
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'loss': ckpt_id,
            'epoch': epoch,
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }
        checkpoint_path = f"{args['save_dir']}/checkpoint_{ckpt_id:.2f}.ckpt"
        try:
            torch.save(checkpoint, checkpoint_path)
            logging.info(f'Checkpoint saved at {checkpoint_path}.')
        except Exception as e:
            print(f'Failed to save checkpoint at {checkpoint_path}. Error: {e}')


def load_checkpoint(model, args, optimizer=None, lr_scheduler=None):
    """Load a model checkpoint."""

    if args['deepspeed']:
        # Load checkpoint using DeepSpeed
        checkpoint_name, client_state = model.load_checkpoint(args['load_dir'], args['ckpt_id'])
        if checkpoint_name is None:
            print("No checkpoint found at specified path!")
            step = 0
            epoch = 0
        else:
            step = client_state.get('step', 0)
            epoch = client_state.get('epoch', 0)
    else:
        # Load checkpoint directly using torch.load for non-DeepSpeed case
        checkpoint_path = f"{args['save_dir']}/checkpoint_{args['ckpt_id']:.2f}.ckpt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cuda:0')  # Assuming single GPU at cuda:0
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}!")
            step = 0
            epoch = 0
        else:
            # Load model, optimizer, and lr_scheduler states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            step = checkpoint.get('step', 0)
            epoch = checkpoint.get('epoch', 0)

    return epoch, step


def train(model: Transformer, train_config: TrainArgs, train_dataloader: DataLoader, eval_dataloader: DataLoader,
          args: Dict):
    """
    main training function to train llama2
    Args:
        model: LLama2 transformer.
        train_config: training config class. contains main training params.
        train_dataloader: training dataloader.
        eval_dataloader: evaluation dataloader
        args: Dict containing all combined args.

    Returns:

    """
    optimizer = AdamW(model.parameters(), lr=train_config.lr)
    scheduler = LambdaLR(optimizer=optimizer,
                         lr_lambda=lambda step: rate(
                             step=step,
                             model_size=model.args.dim,
                             warmup=train_config.warmup_steps,
                         ))
    if args['deepspeed']:
        deepspeed.init_distributed()
        logging.info('Deepspeed is enabled.')
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            args=args,
            dist_init_required=False
        )

    # when loading the model, we start training from where we paused using the same epoch and step that
    # training was paused on.
    if args['load_model']:
        start_epoch, start_step = load_checkpoint(model, args, optimizer, scheduler)
    else:
        start_epoch, start_step = 0, 0

    losses = []
    best_eval_loss = float('inf')

    for epoch in tqdm(range(start_epoch, args['n_epochs'])):
        model.train()
        for step, (X, Y) in enumerate(train_dataloader, start=start_step):
            logits, loss = model(X, 0, Y)
            if args['deepspeed']:
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Log every log_interval batches
            if (step + 1) % args['log_interval'] == 0:
                out = estimate_loss(model=model,
                                    eval_iters=args['eval_iters'],
                                    train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader,
                                    device=args['device'])
                losses.extend([out])
                logging.info(
                    f'Epoch: {epoch}, Batch: {step + 1}/{len(train_dataloader)} | train_loss: {out["train"]:.2f}, '
                    f'eval_loss: {out["eval"]:.2f}')

            # save the model if it was outperforming the previous best model
            cur_eval_loss = losses[-1]['eval']
            if cur_eval_loss < best_eval_loss and step % args['save_interval'] == 0:
                ckpt_id = loss.item()
                save_ds_checkpoint(step, epoch, model, ckpt_id, args, optimizer, scheduler)
                logging.info(f"New best model saved with eval_loss: {cur_eval_loss:.2f}")
    df = pd.DataFrame(losses)
    df.to_pickle(args['save_dir'] + '/losses.pkl')
    return losses
