import os

import yaml
from dotenv import load_dotenv
from flexparse import SUPPRESS, ArgumentParser, IntRange, create_action

from configs import DataConfigs
from core.preprocess import UttutPreprocessor
from core.preprocess.config_objects import CorpusConfig, LanguageConfig
from core.preprocess.record_objects import MetaData, TextDataset
from library.utils import NamedDict, format_id, format_path


load_dotenv('.env')

CONFIG_PATH = 'datasets/corpus.yaml'
LANGUAGE_CONFIGS = {
    'english': LanguageConfig(
        embedding_path=os.getenv('PRETRAINED_EN_WORD_FASTTEXT_PATH'),
        split_token=' ',
    ),
    'test': LanguageConfig(
        embedding_path='datasets/en_fasttext_word2vec_V100D20.msg',
        split_token=' ',
    ),
}


def preprocess(args: DataConfigs) -> tuple[dict[str, TextDataset], MetaData]:
    print(f"data_id: {format_id(args.dataset)}")
    print(f"preprocessor_id {format_id('uttut')}")
    preprocessor = UttutPreprocessor(maxlen=args.maxlen, vocab_size=args.vocab_size)
    corpus_config = load_corpus_table(CONFIG_PATH)[args.dataset]
    return preprocessor.preprocess(corpus_config, return_meta=True)


def load_corpus_table(path):
    corpus_table = NamedDict()
    with open(path) as f:
        for data_id, corpus_dict in yaml.load(f, Loader=yaml.FullLoader).items():
            config = parse_config(corpus_dict)
            if 'train' in config.path and all(os.path.isfile(p) for p in config.path.values()):
                # TODO else warning?
                corpus_table[data_id] = config

    return corpus_table


def parse_config(corpus_dict):
    if isinstance(corpus_dict['path'], dict):
        path = corpus_dict['path']
    else:
        path = {'train': corpus_dict['path']}

    language_id = corpus_dict['language']
    return CorpusConfig(
        path=path,
        language_config=LANGUAGE_CONFIGS[language_id],
        maxlen=corpus_dict.get('maxlen'),
        vocab_size=corpus_dict.get('vocab_size'),
    )


ARGS = [
    create_action(
        '--dataset',
        required=True,
        default=SUPPRESS,
        help='the choice of corpus.',
    ),
    create_action(
        '--maxlen',
        type=IntRange(minval=1),
        help="the max length of sequence padding. "
             f"(use the value declared in {format_path(CONFIG_PATH)} if not given)",
    ),
    create_action(
        '--vocab-size',
        type=IntRange(minval=1),
        help="the maximum number of tokens. ordered by descending frequency. "
             f"(use the value declared in {format_path(CONFIG_PATH)} if not given)",
    ),
]

PARSER = ArgumentParser(add_help=False)
PARSER.add_argument_group(
    'data',
    description="data corpus and preprocessing configurations.",
    actions=ARGS,
)
