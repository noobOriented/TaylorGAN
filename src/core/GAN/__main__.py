from core.train import __main__ as train_main
from library.utils import parse_args_as

from ._trainer_factory import GANObjectiveConfigs


def main():
    configs = parse_args_as(GANTrainingConfigs)
    train_main.main(configs)


class GANTrainingConfigs(train_main.CommonTrainingConfigs, GANObjectiveConfigs, extra='forbid'):
    pass
