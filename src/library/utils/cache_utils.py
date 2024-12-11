import contextlib
import functools
import os
import pickle
import typing as t
from unittest.mock import patch

import numpy as np


class FileCache:

    @classmethod
    def tofile(cls, path, makedirs: bool = True):
        def decorator(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                path_str = cls._get_path_str(path, *args, **kwargs)
                if os.path.isfile(path_str):
                    print(f"Load from {path_str}")
                    return cls.load_data(path_str)

                output = func(*args, **kwargs)
                if makedirs:
                    os.makedirs(os.path.dirname(path_str), exist_ok=True)

                print(f"Cache to {path_str}")
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


def cache_method_call(obj, method: str):
    old_method = getattr(obj, method)
    new_method = functools.cache(old_method)
    return patch.object(obj, method, new_method)
