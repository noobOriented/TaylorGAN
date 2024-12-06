import torch
from flexparse import LookUpCall
from torch.nn import Embedding, Linear
import typing as t
from core.models import Discriminator
from core.preprocess.record_objects import MetaData
from library.torch_zoo.nn import LambdaModule, activations
from library.torch_zoo.nn.masking import (
    MaskAvgPool1d, MaskConv1d, MaskGlobalAvgPool1d, MaskSequential,
)
from library.torch_zoo.nn.resnet import ResBlock
from library.utils import ArgumentBinder, NamedObject


class _DArgs(t.Protocol):
    discriminator: str
    d_fix_embeddings: bool


def create(args: _DArgs, metadata: MetaData) -> Discriminator:
    print(f"Create discriminator: {args.discriminator}")
    network_func = _D_MODELS(args.discriminator)
    embedder = Embedding.from_pretrained(
        torch.from_numpy(metadata.load_pretrained_embeddings()),
        freeze=args.d_fix_embeddings,
    )

    return NamedObject(
        Discriminator(
            network=network_func(embedder.embedding_dim),
            embedder=embedder,
        ),
        name=network_func.argument_info.func_name,
    )


def cnn(input_size, activation: activations.TYPE_HINT = 'relu'):
    ActivationLayer = activations.deserialize(activation)
    return MaskSequential(
        LambdaModule(lambda x: torch.transpose(x, 1, 2)),
        MaskConv1d(input_size, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskConv1d(512, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskAvgPool1d(kernel_size=2),
        MaskConv1d(512, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskConv1d(1024, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskGlobalAvgPool1d(),
        Linear(1024, 1024),
        ActivationLayer(),
    )


def resnet(input_size, activation: activations.TYPE_HINT = 'relu'):
    ActivationLayer = activations.deserialize(activation)
    return MaskSequential(
        Linear(input_size, 512),
        ActivationLayer(),
        LambdaModule(lambda x: torch.transpose(x, 1, 2)),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        MaskGlobalAvgPool1d(),
        Linear(512, 512),
        ActivationLayer(),
    )


_D_MODELS = LookUpCall(
    {
        key: ArgumentBinder(func, preserved=['input_size'])
        for key, func in [
            ('cnn', cnn),
            ('resnet', resnet),
            ('test', lambda input_size: MaskGlobalAvgPool1d(dim=1)),
        ]
    },
    set_info=True,
)
