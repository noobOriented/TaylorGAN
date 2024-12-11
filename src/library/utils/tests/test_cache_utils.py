from unittest.mock import Mock

import numpy as np
import pytest

from ..cache_utils import NumpyCache, PickleCache, cache_method_call


@pytest.mark.parametrize(
    'cacher, output, filename',
    [
        (PickleCache, {'a': 1, 'b': 2, 'c': [3, 4]}, 'test.pkl'),
        (NumpyCache, np.random.choice(100, size=[100]), 'test.npz'),
    ],
)
def test_cache_static(tmpdir, cacher, output, filename):
    filename = tmpdir / filename
    create = Mock(return_value=output)

    @cacher.tofile(filename)
    def wrapped_create():
        return create()

    assert equal(wrapped_create(), output)
    assert filename.isfile()  # save to file
    assert create.call_count == 1

    assert equal(wrapped_create(), output)
    assert create.call_count == 1  # load from file, don't create again


def test_cache_format(tmpdir):
    output = '123'
    create = Mock(return_value=output)

    @PickleCache.tofile(tmpdir / "({0}, {1})")
    def wrapped_create(a, b):
        return create()

    assert equal(wrapped_create(1, 2), output)
    assert (tmpdir / "(1, 2)").isfile()  # save to file
    assert create.call_count == 1

    assert equal(wrapped_create(1, 2), output)
    assert create.call_count == 1  # load from file, don't create again
    assert equal(wrapped_create(2, 1), output)
    assert create.call_count == 2  # different key, create again


def test_cache_callable_path(tmpdir):
    output = '123'
    create = Mock(return_value=output)

    @PickleCache.tofile(path=lambda key: tmpdir / key)
    def wrapped_create(key):
        return create()

    assert wrapped_create('a.pkl') == output
    assert (tmpdir / 'a.pkl').isfile()  # save to file
    assert create.call_count == 1

    assert wrapped_create('a.pkl') == output
    assert create.call_count == 1  # load from file, don't create again

    assert wrapped_create('b.pkl') == output
    assert create.call_count == 2  # different key, create again


def test_cache_method_call():
    obj = Mock(foo=Mock(return_value='123'))
    wrapped_foo = obj.foo
    with cache_method_call(obj, 'foo'):
        assert obj.foo() == '123'
        assert wrapped_foo.call_count == 1
        assert obj.foo() == '123'
        assert wrapped_foo.call_count == 1

    assert obj.foo() == '123'
    assert wrapped_foo.call_count == 2  # no cache


def equal(x, y):
    return np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
