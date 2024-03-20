import sentencepiece as spm

from config import DataArgs
from data import TextFileDataset, create_dataloaders

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(DataArgs.tokenizer_model_path)
dataset = TextFileDataset(DataArgs.filepath, tokenizer, DataArgs.max_seq_length, DataArgs.device, tokenizer.pad_id())

train_dataloader, eval_dataloader = create_dataloaders(dataset, DataArgs.train_size, eval_dataset,
                                                       DataArgs.max_batch_size)
