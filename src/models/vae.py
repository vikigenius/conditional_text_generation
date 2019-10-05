#!/usr/bin/env python

from typing import Dict, List
import numpy
import torch
import logging
import allennlp.nn.util as nn_util

from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import BLEU, Average
from pyro.distributions.torch import Normal
from torch.distributions.kl import kl_divergence
from src.modules.annealer import LossWeight
from src.modules.encoders import VariationalEncoder
from src.modules.decoders.decoder import Decoder

logger = logging.getLogger(__name__)


@Model.register("vae")
class VAE(Model):
    """
    This ``VAE`` class is a :class:`Model` which implements a simple VAE as first described
    in https://arxiv.org/pdf/1511.06349.pdf (Bowman et al., 2015).
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    variational_encoder : ``VariationalEncoder``, required
        The encoder model of which to pass the source tokens
    decoder : ``Model``, required
        The variational decoder model of which to pass the the latent variable
    latent_dim : ``int``, required
        The dimention of the latent, z vector. This is not necessarily the same size as the encoder
        output dim
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 variational_encoder: VariationalEncoder,
                 decoder: Decoder,
                 kl_weight: LossWeight,
                 temperature: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(VAE, self).__init__(vocab)

        self._encoder = variational_encoder
        self._decoder = decoder

        self._latent_dim = variational_encoder.latent_dim

        self._encoder_output_dim = self._encoder.get_encoder_output_dim()

        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token)  # pylint: disable=protected-access
        self._bleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})

        self._kl_metric = Average()
        self.kl_weight = kl_weight

        self._temperature = temperature
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make forward pass for both training/validation/test time.
        """
        encoder_outs = self._encoder(source_tokens)
        p_z = encoder_outs['prior']
        q_z = encoder_outs['posterior']
        kl_weight = self.kl_weight.get()
        if self.training:
            z = q_z.rsample()
            self.kl_weight.step()
        else:
            z = self._encoder.reparametrize(p_z, q_z, self._temperature)

        batch_size = z.size(0)
        kld = kl_divergence(q_z, p_z).sum()/batch_size
        self._kl_metric(kld)
        output_dict = {'z': z, 'predictions': source_tokens['tokens']}

        if not target_tokens:
            return output_dict

        # Do Decoding
        output_dict.update(self._decoder(z, target_tokens))
        rec_loss = output_dict['loss']

        kl_loss = kld*kl_weight
        output_dict['loss'] = rec_loss + kl_loss

        if not self.training:
            best_predictions = output_dict["predictions"]
            self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        all_metrics.update({'klw': float(self.kl_weight.get())})
        all_metrics.update({'kl': float(self._kl_metric.get_metric(reset=reset))})
        return all_metrics

    def generate(self, num_to_sample: int = 1):
        cuda_device = self._get_prediction_device()
        prior_mean = nn_util.move_to_device(torch.zeros((num_to_sample, self._latent_dim)), cuda_device)
        prior_stddev = torch.ones_like(prior_mean)
        prior = Normal(prior_mean, prior_stddev)
        latent = prior.sample()
        generated = self._decoder.generate(latent)

        return self.decode(generated)

    def decode_predictions(self, predictions: torch.Tensor) -> List[str]:
        predicted_indices = predictions

        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_sentences = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]

            # Collect indices after the first start symbol
            if self._start_index in indices:
                indices = indices[indices.index(self._start_index) + 1:]

            predicted_tokens = [self.vocab.get_token_from_index(x) for x in indices]
            all_predicted_sentences.append(predicted_tokens)
        return all_predicted_sentences

    # simple_seq2seq's decode
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        output_dict["predicted_sentences"] = self.decode_predictions(output_dict['predictions'])
        return output_dict
