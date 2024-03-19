from typing import List, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


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

        # Pad the sequence to the max sequence length
        padding_length = self.max_sequence_length - len(tokenized_line)
        if padding_length > 0:
            tokenized_line.extend([self.pad_token_id] * padding_length)

        return torch.tensor(tokenized_line, dtype=torch.long, device=self.device)
