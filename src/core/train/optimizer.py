import typing as t

import more_itertools
import torch

from library.utils import wraps_with_new_signature


def add_custom_optimizer_args(optimizer_cls: type[torch.optim.Optimizer]) -> t.Callable[..., torch.optim.Optimizer]:

    @wraps_with_new_signature(optimizer_cls)
    def factory(*args, clip_norm: float = 0, **kwargs):
        optimizer = optimizer_cls(*args, **kwargs)
        @optimizer.register_step_pre_hook
        def _(*_):
            params = more_itertools.flatten(
                g['params'] for g in optimizer.param_groups
            )
            torch.nn.utils.clip_grad_norm_(params, clip_norm)

        return optimizer

    return factory
