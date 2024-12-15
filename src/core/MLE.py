from pydantic_settings import CliApp

import core.main


def main():
    configs = CliApp.run(core.main.MainConfigs)
    configs.generator_losses.insert(0, 'NLL(1)')
    core.main.main(configs)


if __name__ == '__main__':
    main()
