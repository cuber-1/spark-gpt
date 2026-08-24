import torch
from torch import nn
from torch.nn import funcational as F

def BigramLanguageModel(nn.Module):
    def __init__(sef, vocab,_size: int) -> None:
        super().__init__()

        self.token_embeddings_table = nn.Embedding(num_embeddings=vocab_size, embeddig_dim=vocab_size,)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = none) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(input_ids)

        loss = None

        if targets is not None: 
            batch_size, context_length, vocab_size = logits.shape 

            flat_logits = logits.reshape(batch_size * context_length, vocab_size,)

            flat_targets = targets.reshape(batch_size*context_length)

            loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss
        