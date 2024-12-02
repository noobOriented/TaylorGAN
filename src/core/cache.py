import functools
import os
import pathlib
import warnings

from dotenv import load_dotenv

from library.utils import JSONCache, NumpyCache, PickleCache


class CacheCenter:

    def __init__(self, root_path: str | os.PathLike[str] | None):
        self.root_path = root_path

    def to_file(self, *path, cacher):
        if self.root_path is not None and all(path):
            return cacher.tofile(os.path.join(self.root_path, *path))
        return self._null_decorator

    to_npz = functools.partialmethod(to_file, cacher=NumpyCache)
    to_pkl = functools.partialmethod(to_file, cacher=PickleCache)
    to_json = functools.partialmethod(to_file, cacher=JSONCache)

    @staticmethod
    def _null_decorator(func):
        return func


load_dotenv('.env')
cache_root_dir = pathlib.Path(__file__).parent.parent / '.cache'
cache_root_dir.mkdir(exist_ok=True)
cache_center = CacheCenter(cache_root_dir)
