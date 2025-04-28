from __future__ import annotations

import functools
import typing as t

import pydantic
import torch

from library.utils import LookUpCall
from preprocess import PreprocessResult

from ._generator import AutoRegressiveGenerator, Generator


class GeneratorConfigs(pydantic.BaseModel):
    generator: t.Annotated[
        GeneratorCellFactory,
        pydantic.Field(validation_alias=pydantic.AliasChoices('g', 'generator')),
    ] = 'gru'
    tie_embeddings: t.Annotated[
        bool,
        pydantic.Field(description="whether to tie the weights of generator's input/presoftmax embeddings."),
    ] = False
    g_fix_embeddings: bool = False

    def get_generator(self, data: PreprocessResult) -> Generator:
        print(f"Create generator: {self.generator}")

        embedder = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(data.embedding_matrix),
            padding_idx=data.special_tokens.PAD.idx,
            freeze=self.g_fix_embeddings,
        )
        presoftmax_layer = torch.nn.Linear(embedder.embedding_dim, embedder.num_embeddings)
        if self.tie_embeddings:
            presoftmax_layer.weight = embedder.weight
        else:
            presoftmax_layer.weight.data.copy_(embedder.weight)

        cell = self.generator(embedder.embedding_dim)
        return AutoRegressiveGenerator(
            cell=cell,
            embedder=embedder,
            output_layer=torch.nn.Sequential(
                torch.nn.Linear(
                    cell.hidden_size, 
                    embedder.embedding_dim,
                    bias=False,
                ),
                presoftmax_layer,
            ),
            special_tokens=data.special_tokens,
        )


_G_MODELS = LookUpCall({
    'gru': lambda hidden_size=1024: functools.partial(torch.nn.GRUCell, hidden_size=hidden_size),
    'test': lambda hidden_size=10: functools.partial(torch.nn.GRUCell, hidden_size=hidden_size),
})
GeneratorCellFactory = t.Annotated[
    t.Callable[[int], torch.nn.Module],
    pydantic.PlainValidator(_G_MODELS.parse),
]
GeneratorConfigs.model_rebuild()
