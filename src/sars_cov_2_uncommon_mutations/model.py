from dataclasses import dataclass

import torch

import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    d_embed: int = 10
    context_size: int = 2
    n_hidden: int = 64
    batch_size: int = 32
    n_head: int = 2


class FlattenEmbedding(nn.Module):
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        b, *_ = embeddings.shape
        return embeddings.view(b, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_embed, 3 * config.d_embed)
        self.projection = nn.Linear(config.d_embed, config.d_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape
        q, k, v = self.qkv(x).split(self.config.d_embed, dim = -1) # B, C, D
        q, k, v = self._multi_head_view(q), self._multi_head_view(k), self._multi_head_view(v) # B, n_heads, C, head_size

        # (B, n_heads, C, head_size) x (B, n_heads, head_size, C) -> (B, n_heads, C, C)
        attention_weights = nn.functional.softmax(
            q @ k.transpose(-2, -1) * (D ** -0.5),
            dim = -1,
        )
        assert attention_weights.shape == (B, self.config.n_head, C, C), f"Expected shape (B, {self.config.n_head}, {C}, {C}), got {attention_weights.shape}"
        # (B, n_heads, C, C) x (B, n_heads, C, head_size) -> (B, n_heads, C, head_size)
        y = attention_weights @ v
        assert y.shape == (B, self.config.n_head, C, D // self.config.n_head), f"Expected shape (B, {self.config.n_head}, {C}, {D // self.config.n_head}), got {y.shape}"
        return self.projection(
            y
            .transpose(1, 2)
            .contiguous()
            .view(B, C, D)
        )
        
    def _multi_head_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(
            x.shape[0],
            x.shape[1],
            self.config.n_head,
            x.shape[2] // self.config.n_head 
        ).transpose(1, 2)
        

class MutationLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.attn_block = MultiHeadAttention(config)
        self.head = nn.Linear(config.d_embed, config.vocab_size)
        self.ln1 = nn.LayerNorm(config.d_embed)
        self.ln2 = nn.LayerNorm(config.d_embed)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(x)
        x = x + self.attn_block(self.ln1(x))
        logits = self.head(self.ln2(x))
        if targets is None:
            return x, None
        loss = nn.functional.cross_entropy(
            logits.view(logits.shape[0] * logits.shape[1], logits.shape[-1]),
            targets.view(-1),
            ignore_index = -1
        )
        return x, loss