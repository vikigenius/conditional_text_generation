#!/usr/bin/env python
import torch
from torch.nn import Linear
from pyro.distributions.torch import Normal
from typing import Dict, Tuple
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from src.modules.encoders.variational_encoder import VariationalEncoder


@VariationalEncoder.register('gaussian')
class GaussianEncoder(VariationalEncoder):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 latent_dim: int) -> None:
        super().__init__(text_field_embedder, encoder, latent_dim)
        self._latent_to_mean = Linear(self._encoder.get_output_dim(), self.latent_dim)
        self._latent_to_logvar = Linear(self._encoder.get_output_dim(), self.latent_dim)

    @staticmethod
    def reparametrize(prior: Normal,
                      posterior: Normal,
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Creating the latent vector using the reparameterization trick
        """
        mean = posterior.mean
        std = posterior.stddev
        eps = prior.rsample()
        return eps.mul(std*temperature).add_(mean)

    def forward(self, source_tokens: Dict[str, torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make a forward pass of the encoder, then returning the hidden state.
        """
        final_state = self.encode(source_tokens)
        mean = self._latent_to_mean(final_state)
        logvar = self._latent_to_logvar(final_state)
        prior = Normal(torch.zeros((mean.size(0), self.latent_dim), device=mean.device),
                       torch.ones((mean.size(0), self.latent_dim), device=mean.device))
        posterior = Normal(mean, (0.5 * logvar).exp())
        return {
            'prior': prior,
            'posterior': posterior,
        }
