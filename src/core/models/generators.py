import abc
from itertools import chain

import torch

from core.preprocess import SpecialToken
from library.torch_zoo.functions import random_choice_by_logits

from .interfaces import ModuleInterface
from .sequence_modeling import SampledTokenSequence, TokenSequence


class Generator(torch.nn.Module, ModuleInterface):

    scope = 'Generator'

    def __init__(self, embedder: torch.nn.Module, special_tokens: type[SpecialToken]):
        super().__init__()
        self.embedder = embedder
        self.special_tokens = special_tokens

    @abc.abstractmethod
    def generate(self, batch_size: int, maxlen: int, temperature: float | None = None) -> TokenSequence:
        ...

    @property
    def embedding_matrix(self):
        return self.embedder.weight


class AutoRegressiveGenerator(Generator):

    def __init__(
        self,
        cell: torch.nn.Module,
        embedder: torch.nn.Module,
        output_layer: torch.nn.Module,
        special_tokens: type[SpecialToken],
    ):
        super().__init__(embedder, special_tokens)
        self.cell = cell
        self.output_layer = output_layer

    def generate(
        self,
        batch_size: int,
        maxlen: int,
        temperature: float | None = None,
    ) -> SampledTokenSequence:
        word_idx, state = self._get_start_token_and_state(batch_size)
        logits_list, ids_list, gv_list = [], [], []

        for _ in range(maxlen):
            word_logits, state = self._step_func(word_idx, state)
            if temperature is not None:
                word_logits /= temperature
            word_idx, gv = random_choice_by_logits(word_logits, return_gumbel=True)

            logits_list.append(word_logits)
            ids_list.append(word_idx)
            gv_list.append(gv)

        return SampledTokenSequence(
            logits=torch.stack(logits_list, dim=1),
            ids=torch.stack(ids_list, dim=1),
            gumbel_vars=torch.stack(gv_list, dim=1),
            eos_idx=self.special_tokens.EOS.idx,
            pad_idx=self.special_tokens.PAD.idx,
        )

    def forward(self, batch_size, maxlen, temperature=None):
        return self.generate(batch_size, maxlen, temperature).ids

    def seq_neg_logprobs(self, word_ids: torch.Tensor) -> torch.Tensor:
        word_ids = word_ids.type(torch.int64)
        sos_idx, state = self._get_start_token_and_state(batch_size=word_ids.shape[0])
        logits_list = []
        for word_idx in chain([sos_idx], torch.unbind(word_ids, dim=1)[:-1]):
            word_logits, state = self._step_func(word_idx, state)
            logits_list.append(word_logits)

        return SampledTokenSequence(
            logits=torch.stack(logits_list, dim=1),
            ids=word_ids,
            eos_idx=self.special_tokens.EOS.idx,
            pad_idx=self.special_tokens.PAD.idx,
        ).seq_neg_logprobs

    def _get_start_token_and_state(self, batch_size):
        sos_idx = torch.full([batch_size], self.special_tokens.SOS.idx)
        state = torch.zeros([batch_size, self.cell.hidden_size])
        return sos_idx, state

    def _step_func(self, word_idx, state):
        word_vec = self.embedder(word_idx)
        # TODO LSTM
        output = state = self.cell(word_vec, state)  # output shape (N, C)
        word_logits = self.output_layer(output)
        return word_logits, state

    @property
    def networks(self):
        return [self.cell, self.embedder, self.output_layer]
