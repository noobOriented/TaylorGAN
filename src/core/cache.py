import functools
import os
import pathlib

from library.utils import NumpyCache, PickleCache


class CacheCenter:

    def __init__(self, root_path: str | os.PathLike[str]):
        self.root_path = pathlib.Path(root_path)

    def to_file(self, *path, cacher):
        return cacher.tofile(os.path.join(self.root_path, *path))

    to_npz = functools.partialmethod(to_file, cacher=NumpyCache)
    to_pkl = functools.partialmethod(to_file, cacher=PickleCache)


cache_root_dir = pathlib.Path(__file__).parents[2] / '.cache'
cache_root_dir.mkdir(exist_ok=True)
cache_center = CacheCenter(cache_root_dir)
