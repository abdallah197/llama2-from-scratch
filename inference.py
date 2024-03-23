import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from config import ModelArgs
from model import Transformer

parser = argparse.ArgumentParser(description="Run inference with multiple prompts.")
parser.add_argument('prompts', nargs='+', help='List of prompts for inference')
args = parser.parse_args()

class LLama:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path: str, load_model: bool,
              max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()

        # load the checkpoint of the model
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0
            chk_path = checkpoints[0]
            print(f'loading checkpoint {chk_path}')
            checkpoint = torch.load(chk_path)
            print(f'Loaded checkpoint in {(time.time() - prev_time):.2f} seconds')
            prev_time = time.time()
        if os.path.exists(Path(checkpoint_dir) / 'params.json'):
            with open(Path(checkpoint_dir) / 'params.json', 'r') as f:
                params = json.loads(f.read())
        else:
            params = {}

        model_args: ModelArgs = ModelArgs(
            max_batch_size=max_batch_size,
            device=device,
            max_seq_length=max_seq_len,
            **params
        )

        ## load the tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # set the tensor type as instructed in the paper
        # if we use GPU, we change the precision to  16-bit half-precision floating-point numbers (also known
        # as float16 or half) on CUDA-enabled GPUs.
        if device == 'cuda':
            torch.set_default_dtype(torch.float16)  # Set default to half precision for CUDA
        else:
            torch.set_default_dtype(torch.bfloat16)  # Set default to bfloat16 for other devices

        model = Transformer(model_args).to(device)

        if load_model:
            # we don't need to load the Rope embeddings
            if 'rope.freqs' in checkpoint:
                del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {(time.time() - prev_time):.2f}')
        return LLama(model, tokenizer, model_args)

    def generate(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9,
                 max_gen_len: Optional[int] = None):
        """
        Generate text based on a list of prompt strings.

        Args:
            prompts (List[str]): List of prompt strings for text generation.
            temperature (float): Controls the randomness in the generation process. Lower values make the model more deterministic.
            top_p (float): Controls nucleus sampling. Only the top p percent of tokens are considered for sampling.
            max_gen_len (Optional[int]): Maximum length of the generated text. If None, defaults to the model's maximum sequence length minus 1.

        Returns:
            Tuple[List[List[int]], List[str]]: A tuple containing two lists. The first list contains the tokenized version of the generated texts, and the second list contains the generated texts in string format.
        """
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_length - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, (f"batch size must be less than"
                                                        f" or equal to {self.args.max_batch_size}")
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_length, (f"prompt length must be less than"
                                                         f" or equal to {self.args.max_seq_length}")
        total_len = min(self.args.max_seq_length, max_prompt_len + max_gen_len)

        # create the list that contain the generated tokens, along with the prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")

        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits, _ = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)
            if temperature > 0:
                # the temperature is applied before the softmax. we only take the last token. hence the logits[:, -1]
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy select the next token.
                next_token = torch.argmax(logits[:, -1], dim=-1)
            # The reshape(-1) operation is a safeguard to ensure that next_token is a 1-dimensional tensor with its
            # length equal to the batch size, regardless of how the sampling function returns the values.
            next_token = next_token.reshape(-1)

            # only replace the token if it's a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            # if all prompts in the batch size reached an eos, we stop the for loop
            if all(eos_reached):
                break
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text

    def _sample_top_p(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Sample a token index from probability distribution of tokens with cumulative probability <= top_p.

        Args:
            probs (torch.Tensor): The original probabilities of tokens (shape: [batch_size, vocab_size]).
            top_p (float): The cumulative probability threshold. Tokens with cumulative probabilities
                           greater than top_p are not considered for sampling.

        Returns:
            torch.Tensor: Indices of the sampled tokens (shape: [batch_size, 1]).
        """
        # Sort the probabilities in descending order and also get the original indices
        probs_sorted, indices = torch.sort(probs, dim=-1, descending=True)

        # Compute the cumulative sum of the sorted probabilities
        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)

        # Create a mask to zero out probabilities that are beyond the top_p threshold
        # Subtracting probs_sorted shifts the cumulative sums to the right, ensuring that exactly
        # top_p of cumulative probabilities are retained
        mask = cumulative_probs - probs_sorted > top_p

        # Zero out the probabilities that are not in the top_p
        probs_sorted[mask] = 0.0

        # Normalize the modified probabilities to ensure their sum equals 1
        probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)

        # Sample a token from the modified distribution
        sampled_token = torch.multinomial(probs_sorted, num_samples=1)

        # Map the sampled token index back to the original index in the vocabulary
        final_token = torch.gather(indices, -1, sampled_token)

        return final_token


if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    model = LLama.build(
        checkpoint_dir='llama-2-7b',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=10,
        max_batch_size=len(args.prompts),
        device=device
    )
    out_tokens, out_text = model.generate(args.prompts, max_gen_len=100)
    print(out_text)
    print('All Ok')
