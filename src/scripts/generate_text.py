import argparse
import pathlib

from core.evaluate import TextGenerator
from core.preprocess import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        type=pathlib.Path,
        required=True,
        help='path of serving model folder.',
    )
    parser.add_argument(
        '--export-path',
        type=pathlib.Path,
        required=True,
        help='path to save generated texts.',
    )
    parser.add_argument(
        '--samples',
        type=int,  # IntRange(minval=1),
        default=10000,
        help='number of generated samples(sentences).',
    )
    args = parser.parse_args()

    with open(args.model_path.parent / 'tokenizer.json') as f:
        tokenizer = Tokenizer.model_validate_json(f.read())

    generator = TextGenerator.load_traced(args.model_path, tokenizer=tokenizer)
    print(f"Generate sentences to '{args.export_path}'")
    with open(args.export_path, 'w') as f_out:
        f_out.writelines([
            sentence + "\n"
            for sentence in generator.generate_texts(args.samples)
        ])


if __name__ == '__main__':
    main()
