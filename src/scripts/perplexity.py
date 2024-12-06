import pathlib
import typing as t

import pydantic

from core.evaluate import TextGenerator
from factories import data_factory
from library.utils import parse_args_as


def main():

    class Args(data_factory.DataConfigs):
        model_path: t.Annotated[pathlib.Path, pydantic.Field(description='path of serving model folder.')]

    args = parse_args_as(Args)
    data_collection, meta = args.load_data()
    generator = TextGenerator.load_traced(args.model_path, tokenizer=meta.tokenizer)
    for tag, dataset in data_collection.items():
        print(f"Evaluate {tag} perplexity:")
        perplexity = generator.perplexity(dataset.ids)
        print(f"Perplexity = {perplexity}")


if __name__ == '__main__':
    main()
