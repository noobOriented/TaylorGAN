import typing as t
from functools import partial

import pydantic
import torch
from torch.nn import Embedding, GRUCell, Linear, Sequential

from core.models import AutoRegressiveGenerator, Generator
from core.preprocess import PreprocessResult
from library.utils import LookUpCall


class GeneratorConfigs(pydantic.BaseModel):
    generator: str = 'gru'
    tie_embeddings: t.Annotated[
        bool,
        pydantic.Field(description="whether to tie the weights of generator's input/presoftmax embeddings."),
    ] = False
    g_fix_embeddings: bool = False

    def get_generator(self, data: PreprocessResult) -> Generator:
        print(f"Create generator: {self.generator}")

        embedder = Embedding.from_pretrained(
            torch.from_numpy(data.embedding_matrix),
            padding_idx=data.special_tokens.PAD.idx,
            freeze=self.g_fix_embeddings,
        )
        presoftmax_layer = Linear(embedder.embedding_dim, embedder.num_embeddings)
        if self.tie_embeddings:
            presoftmax_layer.weight = embedder.weight
        else:
            presoftmax_layer.weight.data.copy_(embedder.weight)

        cell = _G_MODELS(self.generator)(embedder.embedding_dim)
        return AutoRegressiveGenerator(
            cell=cell,
            embedder=embedder,
            output_layer=Sequential(
                Linear(cell.hidden_size, embedder.embedding_dim, bias=False),
                presoftmax_layer,
            ),
            special_tokens=data.special_tokens,
        )


def gru_cell(units: int = 1024):
    return partial(GRUCell, hidden_size=units)


_G_MODELS = LookUpCall({
    'gru': gru_cell,
    'test': lambda: partial(GRUCell, hidden_size=10),
})
