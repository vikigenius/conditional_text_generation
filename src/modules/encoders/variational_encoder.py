#!/usr/bin/env python

from typing import Dict, Tuple
import torch
from torch import nn
from allennlp.common import Registrable
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from pyro.distributions import Distribution


class VariationalEncoder(nn.Module, Registrable):
    """
    This ``MaskedEncoder``. This class wraps Pytorch RNN with embedding and masking
    Parameters
    ----------
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2VecEncoder``, required
        The encoder of the "encoder/decoder" model
    latent_dim: The dimension of latent space
    """

    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 latent_dim: int) -> None:
        super().__init__()
        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._latent_dim = latent_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    def get_encoder_input_dim(self) -> int:
        return self._encoder.get_input_dim()

    def get_encoder_output_dim(self) -> int:
        return self._encoder.get_output_dim()

    def encode(self, source_tokens: Dict[str, torch.LongTensor]) -> torch.Tensor:
        embeddings = self._text_field_embedder(source_tokens)
        mask = get_text_field_mask(source_tokens)
        final_state = self._encoder(embeddings, mask)
        return final_state

    @staticmethod
    def reparametrize(prior: Distribution,
                      posterior: Distribution,
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Creating the latent vector using the reparameterization trick
        """
        raise NotImplementedError

    def forward(self, source_tokens: Dict[str, torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make a forward pass of the encoder, then returning the hidden state.
        """
        raise NotImplementedError
