#!/usr/bin/env python3
import torch
from overrides import overrides
from typing import Dict, Optional
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import BooleanAccuracy
from torch.nn import Linear, BCEWithLogitsLoss, Sequential, BatchNorm1d, LeakyReLU


@Model.register("generator")
class Generator(Model):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 activation: Activation = LeakyReLU(0.2),
                 initializer: InitializerApplicator = None):
        super().__init__(None)
        self._latent_mapper = Sequential(
            Linear(input_dim, 2*hidden_dim), BatchNorm1d(2*hidden_dim), activation,
            Linear(2*hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), activation,
            Linear(hidden_dim, output_dim)
        )
        self._ce_loss = BCEWithLogitsLoss()
        self._accuracy = BooleanAccuracy()
        initializer(self)

    @overrides
    def forward(self,
                latent: torch.Tensor,
                discriminator: Optional[Model] = None) -> Dict[str, torch.Tensor]:
        output = self._latent_mapper(latent)
        output_dict = {'generation': output}
        if discriminator is not None:
            dialog = torch.cat([latent, output], dim=-1)
            predicted = discriminator(dialog)["output"]
            # We desire for the discriminator to think this is real.
            desired = torch.ones_like(predicted)
            ce_loss = self._ce_loss(predicted, desired)
            output_dict["loss"] = ce_loss
            self._accuracy(torch.sigmoid(predicted).round(), desired)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"gacc": self._accuracy.get_metric(reset=reset)}
        return metrics
