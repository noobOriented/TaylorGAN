import typing as t
from functools import partialmethod

from core.evaluate import TextGenerator
from library.utils import logging_indent, reuse_method_call

from ..pubsub import Channel
from .base import Callback


class TextEvaluator(Callback):

    def __init__(self, generator: TextGenerator):
        self.on_batch_end = EvaluateCommander(generator)
        self.on_epoch_end = EvaluateCommander(generator)

    def summary(self):
        with logging_indent(self.__class__.__name__):
            for method_name in ('on_batch_end', 'on_epoch_end'):
                with logging_indent(method_name):
                    getattr(self, method_name).summary()


class EvaluateCommander:

    def __init__(self, generator: TextGenerator, /):
        self.generator = generator
        self._command_list = []

    def __call__(self, step, *_):
        with reuse_method_call(self.generator, ['generate_ids', 'generate_texts']) as generator:
            for command in self._command_list:
                command(generator, step)

    def _evaluate(
        self,
        evaluator: t.Callable,
        sample_size: int,
        on_text: bool,
        channel: Channel | None = None,
        period: int = 1,
    ):
        def command(generator: TextGenerator, step):  # late bind generator to allow cache
            if step % period != 0:
                return

            foo = generator.generate_texts if on_text else generator.generate_ids
            samples = foo(sample_size)
            result = evaluator(samples)
            if channel:
                channel.notify(step, result)

        command._info = f"{evaluator.__name__} on {sample_size} samples, every {period} steps"
        self._command_list.append(command)

    def summary(self):
        for command in self._command_list:
            print(command._info)

    evaluate_ids = partialmethod(_evaluate, on_text=False)
    evaluate_texts = partialmethod(_evaluate, on_text=True)
