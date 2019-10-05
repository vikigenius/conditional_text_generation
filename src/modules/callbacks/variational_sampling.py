#!/usr/bin/env python
import logging
import random
from allennlp.data import Instance
from allennlp.training import CallbackTrainer
from allennlp.training.callbacks import Callback, Events, handle_event
from typing import Iterable

logger = logging.getLogger(__name__)


@Callback.register("generate_dialog_samples")
class DialogSampleGen(Callback):
    """
    This callback handles generating of sample dialog
    """
    def __init__(self,
                 validation_data: Iterable[Instance],
                 num_replies: int = 1,
                 num_samples: int = 1):
        self.instances = validation_data
        self.num_samples = num_samples
        self.num_replies = num_replies

    def _display_dialog(self, instance, output_dict):
        query_tokens = [str(token) for token in instance['source_tokens']]
        response_tokens = [str(token) for token in instance['target_tokens']]
        predicted_sentences = output_dict["predicted_sentences"]
        prediction = predicted_sentences
        query = ' '.join(query_tokens[1:-1])
        response = ' '.join(response_tokens[1:-1])
        logger.info(f'{query} -> {prediction} <-> ####[{response}]####')

    @handle_event(Events.VALIDATE, priority=1000)
    def generate_sample(self, trainer: 'CallbackTrainer'):
        logger.info("generating sample dialog")
        trainer.model.eval()
        sample_instances = random.sample(self.instances, self.num_samples)
        output_dicts = trainer.model.forward_on_instances(sample_instances)
        for instance, output_dict in zip(sample_instances, output_dicts):
            self._display_dialog(instance, output_dict)


@Callback.register('prior_sample')
class PriorSampleGen(Callback):
    """
    This callback handles sampling from prior
    """
    def __init__(self):
        pass

    @handle_event(Events.VALIDATE, priority=1000)
    def generate_sample(self, trainer: 'CallbackTrainer'):
        logger.info("Generating sample from prior")
        trainer.model.eval()
        pred_sent = trainer.model.generate()['predicted_sentences']
        logger.info(pred_sent)


@Callback.register("reconstruct_samples")
class ReconstructGen(Callback):
    """
    This callback handles generating of sample reconstruction
    """
    def __init__(self,
                 validation_data: Iterable[Instance]):
        self.instances = validation_data

    def _display_rec(self, instance, output_dict):
        query_tokens = [str(token) for token in instance['source_tokens']]
        predicted_sentences = output_dict["predicted_sentences"]
        prediction = predicted_sentences
        query = ' '.join(query_tokens[1:-1])
        logger.info(f'{query} <-> {prediction}')

    @handle_event(Events.VALIDATE, priority=1000)
    def generate_sample(self, trainer: 'CallbackTrainer'):
        logger.info("generating sample reconstruction")
        trainer.model.eval()
        sample_instances = random.sample(self.instances, 1)
        output_dicts = trainer.model.forward_on_instances(sample_instances)
        for instance, output_dict in zip(sample_instances, output_dicts):
            self._display_rec(instance, output_dict)
