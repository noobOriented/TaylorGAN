from __future__ import annotations

import re
import typing as t
import unicodedata

import more_itertools
import pydantic


class Segmentor(pydantic.BaseModel):
    split_token: str = ' '
    operators: list[Operator]

    def segmentize_text(self, s: str) -> list[str]:
        for op in self.operators:
            s = op(s)
        return s

    def join_text(self, texts: t.Iterable[str]) -> str:
        return self.split_token.join(texts)


class Lower(pydantic.BaseModel):
    type: t.Literal['lower']

    def __call__(self, s: str):
        return s.lower()


class Substitute(pydantic.BaseModel):
    type: t.Literal['sub']
    pattern: str
    repl: str

    def __call__(self, s: str) -> str:
        return re.sub(self.pattern, self.repl, s)


class Strip(pydantic.BaseModel):
    type: t.Literal['strip']
    chars: str | None = None

    def __call__(self, s: str) -> str:
        return s.strip(self.chars)


class SplitEnglish(pydantic.BaseModel):
    type: t.Literal['split-english']

    def __call__(self, s: str):
        orig_tokens = s.split()
        split_tokens = more_itertools.flatten(map(_run_split_on_punc, orig_tokens))
        return ' '.join(split_tokens).split()


Operator = t.Annotated[
    Lower | Substitute | Strip | SplitEnglish,
    pydantic.Field(discriminator='type'),
]


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
