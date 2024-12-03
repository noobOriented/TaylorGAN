from functools import partial
import typing as t

import torch
from flexparse import LookUpCall, create_action
from torch.nn import Embedding, GRUCell, Linear, Sequential

from core.models import AutoRegressiveGenerator, Generator
from core.objectives.regularizers import (
    EmbeddingRegularizer, EntropyRegularizer, LossScaler, SpectralRegularizer,
)
from core.preprocess.record_objects import MetaData
from library.utils import NamedObject


class _GArgs(t.Protocol):
    generator: t.Any
    g_fix_embeddings: bool
    tie_embeddings: bool


def create(args: _GArgs, metadata: MetaData) -> Generator:
    cell_func = _G_MODELS(args.generator)
    print(f"Create generator: {cell_func.argument_info.arg_string}")

    embedding_matrix = torch.from_numpy(metadata.load_pretrained_embeddings())
    embedder = Embedding.from_pretrained(embedding_matrix, freeze=args.g_fix_embeddings)
    presoftmax_layer = Linear(embedder.embedding_dim, embedder.num_embeddings)
    if args.tie_embeddings:
        presoftmax_layer.weight = embedder.weight
    else:
        presoftmax_layer.weight.data.copy_(embedder.weight)

    cell = cell_func(input_size=embedder.embedding_dim)
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
MODEL_ARGS = [
    create_action(
        '-g', '--generator',
        default='gru',
        help="custom options and registry: \n" + "\n".join(_G_MODELS.get_helps()) + "\n",
    ),
    create_action(
        '--tie-embeddings',
        action='store_true',
        help="whether to tie the weights of generator's input/presoftmax embeddings.",
    ),
    create_action(
        '--g-fix-embeddings',
        action='store_true',
        help="whether to fix embeddings.",
    ),
]

G_REGS = LookUpCall({
    'spectral': LossScaler.as_constructor(SpectralRegularizer),
    'embedding': LossScaler.as_constructor(EmbeddingRegularizer),
    'entropy': LossScaler.as_constructor(EntropyRegularizer),
})
REGULARIZER_ARG = create_action(
    '--g-regularizers',
    nargs='+',
    metavar="REGULARIZER(*args, **kwargs)",
    default=[],
    help="custom options and registry: \n" + "\n".join(G_REGS.get_helps()) + "\n",
)
