import argparse

from core.evaluate import TextGenerator
from factories import data_factory
from scripts.parsers import load_parser


def main():
    parser = argparse.ArgumentParser(parents=[data_factory.PARSER, load_parser()])
    args = parser.parse_args()
    data_collection, meta = data_factory.preprocess(args)
    generator = TextGenerator.load_traced(args.model_path, tokenizer=meta.tokenizer)
    for tag, dataset in data_collection.items():
        print(f"Evaluate {tag} perplexity:")
        perplexity = generator.perplexity(dataset.ids)
        print(f"Perplexity = {perplexity}")


if __name__ == '__main__':
    main()
