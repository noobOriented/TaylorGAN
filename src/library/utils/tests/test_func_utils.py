import inspect

from ..func_utils import wraps_with_new_signature


def test_wraps_with_new_signature():

    def foo(a, b):
        pass

    @wraps_with_new_signature(foo)
    def wrapper(*args, c, **kwargs):
        pass

    assert list(inspect.signature(wrapper).parameters.keys()) == ['a', 'b', 'c']
