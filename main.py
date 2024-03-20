import sentencepiece as spm

from config import DataArgs, ModelArgs, TrainArgs
from data import TextFileDataset, create_dataloaders
from model import Transformer
from train import train

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(DataArgs.tokenizer_model_path)
dataset = TextFileDataset(DataArgs.filepath, tokenizer, DataArgs.max_seq_length, DataArgs.device, pad_token_id=0)
model_args = ModelArgs()
model_args.vocab_size = tokenizer.vocab_size()
train_dataloader, eval_dataloader = create_dataloaders(dataset, DataArgs.train_size, DataArgs.max_batch_size)

model = Transformer(model_args).to(model_args.device)
train(model=model,
      train_config=TrainArgs(),
      train_dataloader=train_dataloader,
      eval_dataloader=eval_dataloader)
