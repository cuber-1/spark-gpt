import torch 
from torch.utils.data import Dataset

from CharacterTokenizer import CharacterTokenizer

class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: CharacterTokenizer, context_length: int) -> None:
        token_ids = tokenizer.encode(text)

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

        if len(self.tokens) <= self.context_length:
            raise ValueError("Text must be longer than the context length")
    def __len__(self) -> int:
        return len(self.tokens) - self.context_length
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[index: index + self.context_length]
        y = self.tokens[index+1: index + 1 + self.context_length]

        return x, y