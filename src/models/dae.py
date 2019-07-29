#!/usr/bin/env python

from typing import Dict
import numpy
import torch
import logging

from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BLEU, Average
from allennlp.training.callbacks import Callback, Events, handle_event
from allennlp.training import CallbackTrainer
from pyro.distributions.torch import Normal
from src.modules.annealer import LossWeight
from src.modules.encoders.deterministic_encoder import DeterministicEncoder
from src.modules.decoders.decoder import Decoder

logger = logging.getLogger(__name__)


@Model.register("dae")
class DAE(Model):
    """
    This ``DAE`` class is a :class:`Model` which implements a simple DAE
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    deterministic_encoder : ``DeterministicEncoder``, required
        The encoder model of which to pass the source tokens
    decoder : ``Model``, required
        The variational decoder model of which to pass the the latent variable
    latent_dim : ``int``, required
        The dimention of the latent, z vector. This is not necessarily the same size as the encoder
        output dim
    """
    def __init__(self,
                 vocab: Vocabulary,
                 deterministic_encoder: DeterministicEncoder,
                 decoder: Decoder) -> None:
        super(DAE, self).__init__(vocab)

        self._encoder = deterministic_encoder
        self._decoder = decoder

        self._latent_dim = deterministic_encoder.latent_dim

        self._encoder_output_dim = self._encoder.get_encoder_output_dim()

        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token)  # pylint: disable=protected-access
        self._bleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make forward pass for both training/validation/test time.
        """
        encoder_outs = self._encoder(source_tokens)
        z = encoder_outs['latent']

        batch_size = z.size(0)
        output_dict = {'z': z, 'predictions': source_tokens['tokens']}

        if not target_tokens:
            return output_dict

        # Do Decoding
        output_dict.update(self._decoder(target_tokens, z))
        rec_loss = output_dict['loss']

        output_dict['loss'] = rec_loss

        if not self.training:
            best_predictions = output_dict["predictions"]
            self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

    def generate(self, latent: torch.Tensor):
        cuda_device = self._get_prediction_device()
        generated = self._decoder.generate(latent)
        return self.decode(generated)

    @overrides
    # simple_seq2seq's decode
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x) for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict


# @Callback.register("generate_samples")
# class SampleGen(Callback):
#     """
#     This callback handles generating of sample dialog
#     """
#     def __init__(self,
#                  num_samples: int = 1):
#         self.num_samples = num_samples

#     @handle_event(Events.VALIDATE, priority=1000)
#     def generate_sample(self, trainer: 'CallbackTrainer'):
#         logger.info("generating sample dialog")
#         trainer.model.eval()
#         gen_tokens = trainer.model.generate(z)['predicted_tokens'][0]
#         gen_sent = ' '.join(gen_tokens[1:])
#         logger.info(gen_sent)
