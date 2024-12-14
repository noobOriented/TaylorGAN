import core.main
from library.utils import parse_args_as


def main():
    configs = parse_args_as(core.main.MainConfigs)
    configs.g_regularizers.insert(0, 'NLL(1)')
    core.main.main(configs)


if __name__ == '__main__':
    main()
