
import matplotlib.pyplot as plt

from pathlib import Path

import pickle

import torch

import torch.nn as nn

from rich.progress import Progress

from sars_cov_2_uncommon_mutations.model import MutationLM, ModelConfig

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


@torch.no_grad()
def evaluate(model: MutationLM, dataloader: torch.utils.data.DataLoader) -> float:
    model.eval()
    loss = sum(
        model(x, y)[1].detach().cpu().item()
        for x, y in dataloader
    )
    model.train()
    return loss / len(dataloader)


def create_dataset_from_mutations(mutations: Iterable[str], tokenizer: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for mutation in mutations:
        wt, site, mut = mutation[0], mutation[1 : -1], mutation[-1]
        x.append([tokenizer[wt], tokenizer[site]])
        y.append([tokenizer[site], tokenizer[mut]])
    return torch.tensor(x, dtype = torch.int64), torch.tensor(y, dtype = torch.int64)


def create_tokenizer(mutations: Iterable[str]) -> dict[str, int]:
    aa_chars = sorted({x[0].upper() for x in mutations} | {x[-1].upper() for x in mutations})
    site_chars = [str(x) for x in sorted({int(x[1 : -1]) for x in mutations})]
    return {char: i for i, char in enumerate(aa_chars + site_chars)}


def plot_loss_curves(train_loss: list[float], val_loss: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(train_loss)
    ax.plot(val_loss)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend(["Train", "Validation"])
    return fig


def main() -> None:
    torch.manual_seed(15485863)
    with open(DATA_DIR / "unique_mutations.pkl", "rb") as f:
        unique_mutations = pickle.load(f)
    tokenizer = create_tokenizer(unique_mutations)
    reverse_tokenizer = {i: char for char, i in tokenizer.items()}
    x, y = create_dataset_from_mutations(unique_mutations, tokenizer)
    x, y = x.to(DEVICE), y.to(DEVICE)
    dataset = MutationDataset(x, y)
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    model_config = ModelConfig(
        vocab_size = len(tokenizer),
        d_embed = 256,
        context_size = 2,
        batch_size = 32,
        n_head = 2,
    )
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = model_config.batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = model_config.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = model_config.batch_size)
    mutation_lm = MutationLM(model_config).to(DEVICE)
    n_epochs = 5
    lr = 1e-3
    optimizer = torch.optim.AdamW(mutation_lm.parameters(), lr = lr)
    train_loss, val_loss = [], []
    with Progress() as progress:
        train_task = progress.add_task("[cyan]Training", total = len(train_dataloader) * n_epochs)
        for epoch in range(n_epochs):
            for bx, by in train_dataloader:
                logits, loss = mutation_lm(bx, by)
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()
                progress.update(
                    train_task,
                    advance = 1,
                    description = f"[cyan]Training [bold]Epoch {epoch + 1}: [green]{loss.item():.4f}",
                )
                train_loss.append(loss.item())
            if (epoch + 1) % 2 == 0:
                epoch_val_loss = evaluate(mutation_lm, val_dataloader)
                print(f"Epoch {epoch + 1} validation loss: {epoch_val_loss:.4f}")
                val_loss.append(epoch_val_loss)
    test_loss = evaluate(mutation_lm, test_dataloader)
    print(f"Test loss: {test_loss:.4f}")
    output = mutation_lm.generate(torch.tensor([[tokenizer["F"], tokenizer["456"]]]).to(DEVICE), 9)
    print("".join(reverse_tokenizer[i] for i in output[0].tolist()))
    fig = plot_loss_curves(train_loss, val_loss)
    fig.savefig(
        "loss_run.png",
        dpi = 200,
        bbox_inches = "tight",
    )


if __name__ == "__main__":
    raise SystemExit(main())