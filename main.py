import sentencepiece as spm

from arguments import get_args
from config import ModelArgs, TrainArgs
from data import TextFileDataset, create_dataloaders
from model import Transformer
from train import train

args = get_args()
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(args['tokenizer_model_path'])
dataset = TextFileDataset(args['filepath'], tokenizer, args['max_seq_length'], pad_token_id=0)
model_args = ModelArgs()
model_args.vocab_size = tokenizer.vocab_size()
train_dataloader, eval_dataloader = create_dataloaders(dataset, args['train_size'], args['batch_size'])

model = Transformer(model_args)
train(model=model,
      train_config=TrainArgs(),
      train_dataloader=train_dataloader,
      eval_dataloader=eval_dataloader,
      args=args)
