import typing as t
from functools import partial

import pydantic
import torch
from flexparse import LookUpCall
from torch.nn import Embedding, GRUCell, Linear, Sequential

from core.models import AutoRegressiveGenerator, Generator
from core.preprocess.record_objects import MetaData
from library.utils import NamedObject


class GeneratorConfigs(pydantic.BaseModel):
    generator: str = 'gru'
    tie_embeddings: t.Annotated[
        bool,
        pydantic.Field(description="whether to tie the weights of generator's input/presoftmax embeddings."),
    ] = False
    g_fix_embeddings: bool = False

    def get_generator(self, metadata: MetaData) -> Generator:
        print(f"Create generator: {self.generator}")

        embedding_matrix = torch.from_numpy(metadata.load_pretrained_embeddings())
        embedder = Embedding.from_pretrained(embedding_matrix, freeze=self.g_fix_embeddings)
        presoftmax_layer = Linear(embedder.embedding_dim, embedder.num_embeddings)
        if self.tie_embeddings:
            presoftmax_layer.weight = embedder.weight
        else:
            presoftmax_layer.weight.data.copy_(embedder.weight)

        cell_func = _G_MODELS(self.generator)
        cell: torch.nn.Module = cell_func(embedder.embedding_dim)
        return NamedObject(
            AutoRegressiveGenerator(
                cell=cell,
                embedder=embedder,
                output_layer=Sequential(
                    Linear(cell.hidden_size, embedder.embedding_dim, bias=False),
                    presoftmax_layer,
                ),
                special_token_config=metadata.special_token_config,
            ),
            name=cell_func.argument_info.func_name,
        )


def gru_cell(units: int = 1024):
    return partial(GRUCell, hidden_size=units)


_G_MODELS = LookUpCall(
    {
        'gru': gru_cell,
        'test': lambda: partial(GRUCell, hidden_size=10),
    },
    set_info=True,
)
