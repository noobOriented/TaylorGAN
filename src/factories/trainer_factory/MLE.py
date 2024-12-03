from core.objectives import MLEObjective
from core.train import NonParametrizedTrainer

from .trainer_factory import TrainerCreator


class MLECreator(TrainerCreator):

    def create_trainer(self, generator_updater) -> NonParametrizedTrainer:
        return NonParametrizedTrainer(generator_updater)

    @property
    def objective(self):
        return MLEObjective()
