from flexparse import Action, ArgumentParser


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
