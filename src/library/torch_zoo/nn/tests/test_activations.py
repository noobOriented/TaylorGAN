from torch.nn import ELU, LeakyReLU, ReLU

from ..activations import deserialize


def test_deserialize():
    assert deserialize('relu') == ReLU
    assert deserialize('elu') == ELU
    assert deserialize('leakyrelu') == LeakyReLU
