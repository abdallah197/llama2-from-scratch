from typing import List
from typing import Tuple

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, random_split, DataLoader


class Tokenizer:
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()

        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, sentence: str):
        """

        :param sentence: sentence
        :return: list of idx
        """
        return [self.stoi[ch] for ch in sentence]

    def decode(self, idxs: List[int]):
        """

        :param idxs: list of idx
        :return:
        """
        return "".join([self.itos[idx] for idx in idxs])


class TextDataset(Dataset):
    """Dataset for reading lines from a text file and processing for NLP tasks.

    Attributes:
        max_seq_len (int): Maximum length of the tokenized sequences.
        tokenizer: Tokenizer function to encode text lines.
        device (str): The device type where tensors will be allocated.
    """

    def __init__(self, file_path: str, tokenizer: Tokenizer, max_seq_len: int, device: str = 'cpu') -> None:
        """
        Initializes the dataset from a text file.

        Args:
            file_path (str): Path to the text file.
            tokenizer: Tokenizer function to convert text lines into encoded tokens.
            max_seq_len (int): The maximum allowed length for a sequence of tokens.
            device (str): The type of device to store the tensors (e.g., 'cpu', 'cuda').
            pad_token_id (int, optional): The token ID used for padding. Defaults to 0.
        """
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.device = device
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
        self.text_blocks = self.create_text_blocks()

    def create_text_blocks(self):
        text_length = len(self.text)

        # Calculate the number of blocks and create indices for block starts
        num_blocks = (text_length - 1) // self.max_seq_len + 1
        start_indices = torch.arange(0, num_blocks * self.max_seq_len, self.max_seq_len)

        # Slice the text into blocks
        text_blocks = [self.text[start_idx:start_idx + self.max_seq_len] for start_idx in start_indices]

        return text_blocks

    def __len__(self) -> int:
        """Determines the number of items in the dataset.

        Returns:
            int: The total number of lines in the file.
        """
        return len(self.text_blocks)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrieves an item by its index from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            torch.Tensor: A tensor containing the token IDs of the processed text line.
        """
        line = self.text_blocks[index]
        tokenized_line = self.tokenizer.encode(line)

        # Truncate the line if necessary
        tokenized_line = tokenized_line[:self.max_seq_len]
        assert len(tokenized_line) == self.max_seq_len
        return torch.tensor(tokenized_line, dtype=torch.long, device=self.device)


def llama_collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares batch data for model training by creating input and target tensors.

    This function processes a batch of data by separating each sequence into input and target components.
    The input (`x`) omits the last token of each sequence, and the target (`y`) omits the first token,
    effectively creating a shifted version of the sequence for language modeling tasks.

    Args:
        batch (List[torch.Tensor]): A list of sequences, where each sequence is a tensor of token IDs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - `x_batch`: The input tensor for the model, excluding the last token of each sequence.
            - `y_batch`: The target tensor for the model, excluding the first token of each sequence.
    """
    x = [item[:-1] for item in batch]  # Inputs with last token removed
    y = [item[1:] for item in batch]  # Targets with first token removed

    x_batch = torch.stack(x)
    y_batch = torch.stack(y)

    return x_batch, y_batch


class TextFileDataset(Dataset):
    """Dataset for reading lines from a text file and processing for NLP tasks.

    Attributes:
        max_sequence_length (int): Maximum length of the tokenized sequences.
        pad_token_id (int): ID to use for padding tokens.
        tokenizer: Tokenizer function to encode text lines.
        device (str): The device type where tensors will be allocated.
        lines (List[str]): List of lines read from the text file.
    """

    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_sequence_length: int, device: str,
                 pad_token_id: int = 0) -> None:
        """
        Initializes the dataset from a text file.

        Args:
            file_path (str): Path to the text file.
            tokenizer: Tokenizer function to convert text lines into encoded tokens.
            max_sequence_length (int): The maximum allowed length for a sequence of tokens.
            device (str): The type of device to store the tensors (e.g., 'cpu', 'cuda').
            pad_token_id (int, optional): The token ID used for padding. Defaults to 0.
        """
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.device = device

        with open(file_path, 'r', encoding='utf-8') as file:
            self.lines = [line.strip() for line in file]

    def __len__(self) -> int:
        """Determines the number of items in the dataset.

        Returns:
            int: The total number of lines in the file.
        """
        return len(self.lines)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrieves an item by its index from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            torch.Tensor: A tensor containing the token IDs of the processed text line.
        """
        line = self.lines[index]
        tokenized_line = self.tokenizer.encode(line, out_type=int, add_bos=True, add_eos=True)

        # Truncate the line if necessary
        tokenized_line = tokenized_line[:self.max_sequence_length]

        # Pad the sequence to the max sequence length.
        padding_length = self.max_sequence_length - len(tokenized_line)
        if padding_length > 0:
            tokenized_line.extend([self.pad_token_id] * padding_length)
        assert len(tokenized_line) == self.max_sequence_length
        return torch.tensor(tokenized_line, dtype=torch.long, device=self.device)


def split_dataset(dataset: Dataset, train_split_ratio: float) -> Tuple[Dataset, Dataset]:
    """Splits a dataset into training and evaluation subsets.
    Args:
        dataset (Dataset): The dataset to be split.
        train_split_ratio (float): The proportion of the dataset to be used for training.
                                   This should be a decimal between 0 and 1.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing two subsets of the original dataset:
            - The first element is the training dataset.
            - The second element is the evaluation dataset.
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_split_ratio)  # E.g., 80% of the dataset for training
    eval_size = dataset_size - train_size  # The remainder for evaluation
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    return train_dataset, eval_dataset


def create_dataloaders(
        dataset: Dataset,
        train_split_ratio: float,
        batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Creates data loaders for training and evaluation datasets.
    Args:
        dataset (Dataset): The torch dataset class that contain the data
        train_split_ratio (float): The proportion of the dataset to be used for training.
                                   This should be a decimal between 0 and 1.
        batch_size (int): The number of samples to be loaded per batch.
    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the data loaders for the training and evaluation datasets.
    """
    train_dataset, eval_dataset = split_dataset(dataset, train_split_ratio)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=llama_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=llama_collate_fn)

    return train_dataloader, eval_dataloader