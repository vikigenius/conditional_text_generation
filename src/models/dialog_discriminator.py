#!/usr/bin/env python
import torch
from typing import Dict
from allennlp.models.model import Model
from allennlp.nn.activations import Activation
from allennlp.nn.initializers import InitializerApplicator
from torch.nn import Linear, BCEWithLogitsLoss, Sequential, BatchNorm1d, LeakyReLU


@Model.register('dialog-discriminator')
class DialogDiscriminator(Model):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: Activation = LeakyReLU(0.2),
                 initializer: InitializerApplicator = None):
        super().__init__(None)
        self._activation = activation
        self._classifier = Sequential(
            Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), activation,
            Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), activation,
            Linear(hidden_dim, 1)
        )
        self._loss = BCEWithLogitsLoss()

    def forward(self,
                inputs: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        inputs = inputs.squeeze(-1)
        output = self._classifier(inputs)
        output_dict = {"output": output}
        if labels is not None:
            output_dict["loss"] = self._loss(output, labels)
            output_dict["accuracy"] = (torch.sigmoid(output).round() == labels).float().mean()
        return output_dict
