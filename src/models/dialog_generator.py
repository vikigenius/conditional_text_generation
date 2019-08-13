#!/usr/bin/env python
import torch
from torch.nn import Linear, BCEWithLogitsLoss, Sequential, BatchNorm1d, LeakyReLU
from typing import Dict, Optional
from overrides import overrides
from pyro.distributions.torch import Normal
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import BooleanAccuracy
from allennlp.nn.activations import Activation
from src.modules.encoders import GaussianEncoder


@Model.register("dialog-generator")
class DialogGenerator(Model):
    def __init__(self,
                 latent_dim: int,
                 activation: Activation = LeakyReLU(0.2),
                 temperature: int = 1.0,
                 initializer: InitializerApplicator = None):
        super().__init__(None)
        self._latent_mapper = Sequential(
            Linear(latent_dim, 2*latent_dim), BatchNorm1d(2*latent_dim), activation,
            Linear(2*latent_dim, 2*latent_dim), BatchNorm1d(2*latent_dim), activation,
            Linear(2*latent_dim, latent_dim)
        )
        self._temperature = temperature
        self._ce_loss = BCEWithLogitsLoss()
        self._accuracy = BooleanAccuracy()
        initializer(self)

    @overrides
    def forward(self,
                query_prior: Normal,
                query_posterior: Normal,
                discriminator: Optional[Model] = None) -> Dict[str, torch.Tensor]:
        query_mean = query_posterior.mean
        query_logvar = 2*torch.log(query_posterior.stddev)
        if self.training:
            query_mean.requires_grad_()
            query_logvar.requires_grad_()
            temperature = 1.0
        else:
            temperature = self._temperature
        query_latent = GaussianEncoder.reparametrize(query_prior, query_posterior, temperature)
        pred_latent = self._latent_mapper(query_latent)
        output_dict = {}
        pred_dialog = torch.cat((query_latent, pred_latent), dim=-1)
        if discriminator is not None:
            predicted = discriminator(pred_dialog)["output"]
            # We desire for the discriminator to think this is real.
            desired = torch.ones_like(predicted)
            ce_loss = self._ce_loss(predicted, desired)
            output_dict["loss"] = ce_loss
            self._accuracy(torch.sigmoid(predicted).round(), desired)

        output_dict.update({
            'predicted_response': pred_latent,
            'predicted_dialog': pred_dialog,
            'output': pred_dialog,
        })
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"gacc": self._accuracy.get_metric(reset=reset)}
        return metrics
