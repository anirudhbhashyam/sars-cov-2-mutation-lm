import torch

import torch.nn as nn


class FlattenEmbedding(nn.Module):
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        b, *_ = embeddings.shape
        return embeddings.view(b, -1)


class MutationLM(nn.Module):
    def __init__(self, vocab_size: int, d_embed: int = 128, context_size: int = 2, n_hidden: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.flatten = FlattenEmbedding()
        self.head = nn.Sequential(
            nn.Linear(d_embed * context_size, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.flatten(x)
        return self.head(x)