import os
import re
import typing as t
import unicodedata

import more_itertools

from library.utils import JSONSerializableMixin, format_path, logging_indent, tqdm_open

from .adaptors import WordEmbeddingCollection


class LanguageConfig(JSONSerializableMixin):

    def __init__(
        self,
        embedding_path: str | None,
        split_token: str = '',
    ):
        self.embedding_path = embedding_path
        self.split_token = split_token

    def segmentize_text(self, text: str) -> list[str]:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.lower()
        return _tokenize(text)

    def load_pretrained_embeddings_msg(self):
        if not (self.embedding_path and os.path.isfile(self.embedding_path)):
            raise FileNotFoundError(f"invalid embedding_path: {self.embedding_path}")
        print(f"Load pretrained embedding from : {format_path(self.embedding_path)}")
        return WordEmbeddingCollection.load_msg(self.embedding_path)

    def get_config(self):
        return {
            'embedding_path': str(self.embedding_path),
            'split_token': self.split_token,
        }

    @classmethod
    def from_config(cls, config_dict):
        return cls(**config_dict)


class CorpusConfig:

    def __init__(
        self,
        name: str,
        path: str | t.Mapping[str, str],
        language_config: LanguageConfig,
        maxlen: int | None = None,  # used when preprocessor.maxlen = None
        vocab_size: int | None = None,  # used when preprocessor.vocab_size = None
    ):
        if not isinstance(path, t.Mapping):
            path = {'train': path}
        
        self.name = name
        self.path = path
        self.language_config = language_config
        self.maxlen = maxlen
        self.vocab_size = vocab_size

    def iter_train_sentences(self) -> t.Iterator[list[str]]:
        with tqdm_open(self.path['train']) as it:
            for s in it:
                yield self.language_config.segmentize_text(s)


class SpecialTokenConfig:

    class TokenIdxTuple(t.NamedTuple):
        token: str
        idx: int

    def __init__(self, **kwargs: str):
        if not more_itertools.all_unique(kwargs.values()):
            raise KeyError("special tokens conflict.")

        self._token_list = list(kwargs.values())
        self._attrs = {
            key: self.TokenIdxTuple(token, idx)
            for idx, (key, token) in enumerate(kwargs.items())
        }

    @property
    def tokens(self) -> list[str]:
        return self._token_list

    def __getattr__(self, key):
        if key in self._attrs:
            return self._attrs[key]
        raise AttributeError

    def summary(self):
        with logging_indent("Special tokens config:"):
            for key, (token, idx) in self._attrs.items():
                print(f"{key} token: '{token}', index: {idx}.")


def _tokenize(s: str) -> list[str]:
    orig_tokens = s.split()
    split_tokens = more_itertools.flatten(
        _run_split_on_punc(token)
        for token in orig_tokens
    )
    return ' '.join(split_tokens).split()


def _run_split_on_punc(text: str) -> list[str]:
    """
    Recognize punctuations and seperate them into independent tokens.

    E.g.
    1. "abc, cdf" -> ["abc", ",", " ", "cdf"]
    2. "I like apples." -> ["I", "like", "apples", "."]

    """
    start_new_word = True
    output: list[list[str]] = []
    for c in text:
        if _is_punctuation(c):
            output.append([])
            start_new_word = True
        elif start_new_word:
            output.append([])
            start_new_word = False

        output[-1].append(c)

    return [''.join(x) for x in output]


def _is_punctuation(char) -> bool:
    """Checks whether `chars` is a punctuation character.

    We treat all non-letter/number ASCII as punctuation.
    Characters such as "^", "$", and "`" are not in the Unicode
    Punctuation class but we treat them as punctuation anyways, for consistency.

    """
    return ord(char) in _ASCII_PUNCTUATIONS or unicodedata.category(char).startswith('P')


_ASCII_PUNCTUATIONS = {*range(33, 48), *range(58, 65), *range(91, 97), *range(123, 127)}
