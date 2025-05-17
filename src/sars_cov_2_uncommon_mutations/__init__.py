from pathlib import Path

import pickle

import torch

import torch.nn as nn

from rich.progress import Progress

from sars_cov_2_uncommon_mutations.model import MutationLM

from typing import Iterable


DATA_DIR = Path(__file__).parents[2].joinpath("data").resolve()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MutationDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def evaluate(model: MutationLM, dataloader: torch.utils.data.DataLoader) -> float:
    model.eval()
    with torch.no_grad():
        loss_fn = nn.CrossEntropyLoss()
        loss = 0
        for x, y in dataloader:
            y_pred = model(x)
            loss += loss_fn(y_pred, y)
    return loss / len(dataloader)


def create_dataset_from_mutations(unique_mutations: Iterable[str], tokenizer: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for mutation in unique_mutations:
        wt, site, mut = mutation[0], mutation[1 : -1], mutation[-1]
        x.append([tokenizer[wt], tokenizer[site]])
        y.append(tokenizer[mut])
    return torch.tensor(x, dtype = torch.int64), torch.tensor(y, dtype = torch.int64)


def create_tokenizer(mutations: Iterable[str]) -> dict[str, int]:
    aa_chars = sorted({x[0].upper() for x in mutations} | {x[-1].upper() for x in mutations})
    site_chars = [str(x) for x in sorted({int(x[1 : -1]) for x in mutations})]
    return {char: i for i, char in enumerate(aa_chars + site_chars)}


def main() -> None:
    torch.manual_seed(15485863)
    with open(DATA_DIR / "unique_mutations.pkl", "rb") as f:
        unique_mutations = pickle.load(f)
    tokenizer = create_tokenizer(unique_mutations)
    x, y = create_dataset_from_mutations(unique_mutations, tokenizer)
    x, y = x.to(DEVICE), y.to(DEVICE)
    dataset = MutationDataset(x, y)
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)
    mutation_lm = MutationLM(
        vocab_size = len(tokenizer),
        d_embed = 10,
        n_hidden = 64,
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = 500
    lr = 1e-2
    with Progress() as progress:
        train_task = progress.add_task("[cyan] Training", total = len(train_dataloader) * n_epochs)
        for epoch in range(n_epochs):
            for bx, by in train_dataloader:
                logits = mutation_lm(bx)
                loss = loss_fn(logits, by)
                for p in mutation_lm.parameters():
                    p.grad = None
                loss.backward()
                for p in mutation_lm.parameters():
                    p.data += lr * -p.grad
                progress.update(
                    train_task,
                    advance = 1,
                    description = f"[cyan]Training [bold]Epoch {epoch + 1}: [green]{loss.item():.4f}"
                )
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    epoch_val_loss = evaluate(mutation_lm, val_dataloader)
                print(f"Epoch {epoch + 1} validation loss: {epoch_val_loss:.4f}")
                mutation_lm.train()
    test_loss = evaluate(mutation_lm, test_dataloader)
    print(f"Test loss: {test_loss:.4f}")

    
if __name__ == "__main__":
    raise SystemExit(main())
