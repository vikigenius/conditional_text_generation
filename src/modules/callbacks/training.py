#!/usr/bin/env python3
import logging
from enum import Enum, auto
from itertools import cycle
from allennlp.training import CallbackTrainer
from allennlp.training.callbacks import Callback, Events, handle_event

logger = logging.getLogger(__name__)


class GanTrainingStage(Enum):
    GENERATOR = auto()
    DISCRIMINATOR_REAL = auto()
    DISCRIMINATOR_FAKE = auto()


@Callback.register("gan_stage")
class GanStage(Callback):
    def __init__(self):
        super().__init__()
        self.stages = cycle(GanTrainingStage)

    @handle_event(Events.BATCH_START)
    def set_stage(self, trainer):
        stage = next(self.stages)
        trainer.model.stage = stage
        trainer.optimizer.stage = stage


@Callback.register('reset_kl')
class ResetKL(Callback):
    """
    This callback handles resetting of KL
    """
    @handle_event(Events.BATCH_END)
    def reset(self, trainer: 'CallbackTrainer'):
        trainer.model._kl_metric.get_metric(reset=True)
