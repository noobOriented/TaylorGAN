import core.main
from library.utils import parse_args_as

from ._trainer_factory import GANTrainerConfigs


def main():
    configs = parse_args_as(GANMainConfigs)
    core.main.main(configs)


class GANMainConfigs(GANTrainerConfigs, core.main.MainConfigs, extra='forbid'):
    pass
