import json
import os
import pickle
from contextlib import contextmanager
from functools import lru_cache, wraps

import numpy as np
import pydantic

from .format_utils import format_path
from .func_utils import ObjectWrapper


class FileCache:

    @classmethod
    def tofile(cls, path, makedirs: bool = True):
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                path_str = cls._get_path_str(path, *args, **kwargs)
                if os.path.isfile(path_str):
                    print(f"Load from {format_path(path_str)}")
                    return cls.load_data(path_str)

                output = func(*args, **kwargs)
                if makedirs:
                    os.makedirs(os.path.dirname(path_str), exist_ok=True)

                print(f"Cache to {format_path(path_str)}")
                cls.save_data(output, path_str)
                return output

            return wrapped

        return decorator

    @staticmethod
    def _get_path_str(path, *args, **kwargs):
        if callable(path):
            return path(*args, **kwargs)
        else:
            return str(path).format(*args, **kwargs)

    def load_data(path):
        raise NotImplementedError

    def save_data(data, path):
        raise NotImplementedError


class PickleCache(FileCache):

    @staticmethod
    def load_data(path):
        with open(path, 'rb') as f_in:
            return pickle.load(f_in)

    @staticmethod
    def save_data(data, path):
        with open(path, 'wb') as f_out:
            pickle.dump(data, f_out)


class NumpyCache(FileCache):

    @staticmethod
    def load_data(path):
        return np.load(path)['data']

    @staticmethod
    def save_data(data, path):
        np.savez_compressed(path, data=data)


class JSONSerializableMixin:
    _subclass_map: dict[str, type] = {}

    @classmethod
    def __init_subclass__(cls):
        cls._subclass_map[cls.__name__] = cls

    def serialize(self):
        return json.dumps(
            {
                'class_name': self.__class__.__name__,
                'config': pydantic.TypeAdapter(self.__class__).dump_python(self, mode='json'),
            },
            indent=2,
        )

    @classmethod
    def deserialize(cls, data: str):
        params = json.loads(data)
        subclass = cls._subclass_map[params['class_name']]
        if not issubclass(subclass, cls):
            raise ValueError(f'{cls.__name__}.deserialize on non-subclass {subclass.__name__} is forbidden!')
        return pydantic.TypeAdapter(subclass).validate_python(params['config'])


@contextmanager
def reuse_method_call(obj, methods: list[str]):
    wrapped_obj = ObjectWrapper(obj)
    for method_name in methods:
        old_method = getattr(obj, method_name)
        new_method = lru_cache(None)(old_method)
        setattr(wrapped_obj, method_name, new_method)

    yield wrapped_obj

    for method_name in methods:
        getattr(wrapped_obj, method_name).cache_clear()
