from ..collections import ExponentialMovingAverageMeter, counter_or


def test_counter_or():
    assert counter_or([
        {'a': 3, 'b': 1},
        {'a': 2, 'c': 1},
    ]) == {'a': 3, 'b': 1, 'c': 1}


def test_exponential_average_meter():
    meter = ExponentialMovingAverageMeter(decay=0.5)
    assert meter(1.0) == 1.0
    assert meter(2.0) == 1.5
    assert meter(3.0) == 2.25
