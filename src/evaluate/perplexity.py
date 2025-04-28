import pathlib
import typing as t

import pydantic
from pydantic_settings import CliApp

from preprocess import DataConfigs

from .generator_executor import TextGenerator


def main():

    class Args(DataConfigs):
        model_path: t.Annotated[pathlib.Path, pydantic.Field(description='path of serving model folder.')]

    args = CliApp.run(Args)
    preprocessed_result = args.load_data()
    generator = TextGenerator.load_traced(args.model_path, tokenizer=preprocessed_result.tokenizer)
    for tag, dataset in preprocessed_result.dataset.items():
        print(f"Evaluate {tag} perplexity:")
        perplexity = generator.perplexity(dataset.ids)
        print(f"Perplexity = {perplexity}")


if __name__ == '__main__':
    main()
