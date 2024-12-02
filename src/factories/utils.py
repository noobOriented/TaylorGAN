import typing as t

from flexparse import Action, ArgumentParser, create_action


def create_factory_action(
    *args,
    type: t.Callable,  # noqa
    help_prefix: str = '',
    default=None,
    **kwargs,
) -> Action:
    return create_action(
        *args,
        type=type,
        default=default,
        help=(
            help_prefix + "custom options and registry: \n" + "\n".join(type.get_helps()) + "\n"
        ),
        **kwargs,
    )


def parent_parser(
    title: str,
    description: str,
    arguments: list[Action],
    **kwargs,
) -> ArgumentParser:
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group(title, description=description)
    for action in arguments:
        group._add_action(action)

    return parser
