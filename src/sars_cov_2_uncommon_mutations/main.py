import asyncio

import asyncclick as click

from functools import wraps

from pathlib import Path

import polars as pl

from rich.progress import Progress

from sars_cov_2_uncommon_mutations import get_mutations
from sars_cov_2_uncommon_mutations import train

from typing import Any, Callable


DATA_PATH = Path(__file__).parents[2].joinpath("data").resolve()


def coroutine(f: Callable[[Any], Any]) -> Callable[[Any], asyncio.Future]:
    @wraps(f)
    def inner(*args, **kwargs) -> None:
        return f(*args, **kwargs)
    return inner


@click.command("pango-lineage")
@click.option("--variant", default = "XBB.1.5", type = str, help = "The pango-lineage variant.")
@coroutine
async def get_mutation_data(variant: str) -> None:
    response_text = await get_mutations.get_mutations_for_variant(variant)
    click.echo(get_mutations.process_mutation_data(response_text))


@click.command("dataset")
async def create_dataset() -> None:
    variants = pl.read_csv(DATA_PATH.joinpath("lineages.csv"))
    task_data = {}
    done_batch_indices = {
        int(p.stem.split("_")[-1])
        for p in DATA_PATH.glob("mutations_data_*.parquet")
    }

    with Progress() as progress:
        task = progress.add_task("Variant batch: 0", total = len(variants))
        for i, df_batch in enumerate(variants.iter_slices(100)):
            if i in done_batch_indices:
                continue
            tasks = await get_mutations.get_mutations_variant_batch(df_batch["lineage"])
            df_batch_result = pl.concat(
                [
                    get_mutations.process_mutation_data(t.result()) for t in tasks
                ]
            )
            df_batch_result.write_parquet(DATA_PATH / f"mutations_data_{i}.parquet")
            progress.update(task, advance = len(df_batch), description = f"Lineage batch: {i}")


@click.command("train")
def train_model() -> None:
    train.train()


def main() -> int:
    cli = click.Group()
    cli.add_command(get_mutation_data)
    cli.add_command(create_dataset)
    cli.add_command(train_model)
    cli()
    return 0    