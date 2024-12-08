import gc
import pathlib
from unittest.mock import patch

import pytest

from scripts import evaluate_text, generate_text, perplexity, restore_from_checkpoint, train


@pytest.fixture(scope='session')
def cache_root_dir(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp('cache')


@pytest.fixture(scope='session')
def serving_root(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp('tf_serving')


@pytest.fixture(scope='session')
def checkpoint_root(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp('checkpoint')


@pytest.fixture(scope='session', autouse=True)
def redirect_cache_root(cache_root_dir):
    from core.cache import cache_center
    cache_center.root_path = cache_root_dir


@pytest.fixture(autouse=True)
def transact():
    yield
    gc.collect()


class TestTrain:

    @pytest.mark.dependency(name='train_GAN')
    def test_GAN(self, serving_root, checkpoint_root):
        with patch(
            'sys.argv',
            ' '.join([
                '... --data test',
                '--gen test --dis test --estimator taylor',
                '--g-op sgd(1e-3,clip_norm=1) --g-reg embedding(0.1) entropy(1e-5)',
                '--d-op sgd(1e-3,clip_norm=1) --d-reg grad_penalty(10) spectral(0.1) embedding(0.1)',
                '--epochs 4 --batch 2',
                '--bleu 2',
                f'--serv {serving_root} --checkpoint {checkpoint_root} --save-period 2',
            ]).split(),
        ):
            train.GAN_main()

    def test_MLE(self, serving_root, checkpoint_root):
        with patch(
            'sys.argv',
            ' '.join([
                '... --data test',
                '--gen test --g-op sgd(1e-3)',
                '--epochs 4 --batch 2',
                f'--serv {serving_root} --checkpoint {checkpoint_root} --save-period 2',
            ]).split(),
        ):
            train.MLE_main()


class TestSaveLoad:

    @pytest.mark.dependency(name='save_serving', depends=['train_GAN'])
    def test_serving_model_is_saved(self, serving_root: pathlib.Path):
        epochs, period = 4, 2
        model_dir = min(serving_root.iterdir())
        assert (model_dir / 'tokenizer.json').is_file()

        for epo in range(period, epochs, period):
            epo_dirname = model_dir / f'model_epo{epo}.pth'
            assert epo_dirname.is_file()

    @pytest.mark.dependency(name='restore', depends=['train_GAN'])
    def test_restore(self, checkpoint_root: pathlib.Path):
        restore_path = min(checkpoint_root.iterdir())
        with patch('sys.argv', f'... {restore_path} --epochs 6 --save-period 5'.split()):
            restore_from_checkpoint.main()

        # successfully change saving_epochs
        assert {p.name for p in restore_path.iterdir()} == {
            'args',
            'epoch2.pth',
            'epoch4.pth',
            'epoch5.pth',
        }


class TestEvaluate:

    @pytest.mark.dependency(name='generate_text', depends=['save_serving'])
    def test_generate_text(self, tmp_path: pathlib.Path, serving_root: pathlib.Path):
        model_path = min(serving_root.iterdir()) / 'model_epo4.pth'
        export_path = tmp_path / 'generated_text.txt'
        with patch('sys.argv', f'... --model {model_path} --export {export_path} --samples 100'.split()):
            generate_text.main()

        assert len(export_path.read_text().splitlines()) == 100

    @pytest.mark.dependency(name='perplexity', depends=['save_serving'])
    def test_perplexity(self, serving_root: pathlib.Path):
        model_path = min(serving_root.iterdir()) / 'model_epo4.pth'
        with patch('sys.argv', f'... --model {model_path} --data test'.split()):
            perplexity.main()

    def test_evaluate_text(self, data_dir: pathlib.Path):
        corpus_path = data_dir / 'train.txt'
        with patch('sys.argv', f'... --eval {corpus_path} --data test --bleu 5'.split()):
            evaluate_text.main()
