import asyncio

from dataclasses import dataclass

import httpx

import polars as pl

from io import StringIO

from typing import (
    Any,
    Iterable,
)


@dataclass
class MutationEntry:
    mutation: str
    proportion: str
    count: int
    sequenceName: str | None
    mutationFrom: str
    mutationTo: str
    position: int


async def get_mutations_for_variant(variant: str) -> str:
    host = "https://lapis.cov-spectrum.org/open/v2"
    endpoint = "sample/aminoAcidMutations"
    url = "/".join((host, endpoint))
    json_query = {
        "minProportion": "0.05",
        "pangoLineage": variant,
        "dataFormat": "csv",
    }
    query = "&".join(
        f"{k}={v}"
        for k, v in json_query.items()
    )
    full_url = "?".join((url, query))
    async with httpx.AsyncClient(timeout = 400) as client:
        response = await client.get(full_url)
        response.raise_for_status()
    return response.text


def process_mutation_data(response_text: str) -> pl.DataFrame:
    temp_data = StringIO(response_text)
    df = pl.read_csv(
        temp_data,
        schema = {
            "mutation": pl.String,
            "count": pl.Int64,
            "coverage": pl.Int64,
            "proportion": pl.Float64,
            "sequenceName": pl.String,
            "mutationFrom": pl.String,
            "mutationTo": pl.String,
            "position": pl.Int64,
        }
    )
    return df


async def get_mutations_variant_batch(variants: Iterable[str]) -> list[asyncio.Task]:
    async with asyncio.TaskGroup() as tg:
        return [
            tg.create_task(get_mutations_for_variant(v))
            for v in variants
        ]
