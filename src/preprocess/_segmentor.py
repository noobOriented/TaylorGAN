from __future__ import annotations

import re
import typing as t
import unicodedata

import more_itertools
import pydantic


class EnglishSegmentor(pydantic.BaseModel):
    type: t.Literal['english'] = 'english'

    def segmentize_text(self, s: str) -> list[str]:
        s = re.sub(r'\s+', ' ', s).strip().lower()
        orig_tokens = s.split()
        split_tokens = more_itertools.flatten(map(_run_split_on_punc, orig_tokens))
        return ' '.join(split_tokens).split()

    def join_text(self, texts: t.Iterable[str]) -> str:
        return ' '.join(texts)


Segmentor = t.Annotated[EnglishSegmentor, pydantic.Field(discriminator='type')]


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
