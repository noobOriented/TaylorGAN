from pydantic_settings import CliApp

import core.main

from ._trainer_factory import GANTrainerConfigs


def main():
    configs = CliApp.run(GANMainConfigs)
    core.main.main(configs)


class GANMainConfigs(GANTrainerConfigs, core.main.MainConfigs, extra='forbid'):
    pass
