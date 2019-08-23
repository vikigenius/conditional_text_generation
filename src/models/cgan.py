#!/usr/bin/env python

import numpy
import torch
import logging
from torch.nn import Linear, BCEWithLogitsLoss, Sequential, BatchNorm1d, LeakyReLU
from overrides import overrides
from typing import List, Dict, Optional
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import Average

from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import BooleanAccuracy
from allennlp.nn.activations import Activation

from src.modules.encoders import VariationalEncoder
from src.modules.decoders import VariationalDecoder
from src.modules.metrics import NLTKSentenceBLEU
from nltk.translate.bleu_score import SmoothingFunction
from torch.distributions import Normal


logger = logging.getLogger(__name__)


@Model.register("conditional_dialog_gan")
class ConditionalDialogGan(Model):
    """
    Our trainer wants a single model, so we cheat by encapsulating both the
    generator and discriminator inside a single model. We'll access them individually.
    """
    # pylint: disable=abstract-method
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: VariationalEncoder,
                 decoder: VariationalDecoder,
                 generator: Model,
                 discriminator: Model,
                 train_temperature: float = 0.0,
                 inference_temperature: float = 0.0,
                 num_responses: int = 10) -> None:
        super().__init__(vocab)
        self._encoder = encoder
        self._decoder = decoder
        self.train_temperature = train_temperature
        self.inference_temperature = inference_temperature
        self._num_responses = num_responses
        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token)  # pylint: disable=protected-access
        self.s_bleu4 = NLTKSentenceBLEU(n_hyps=self._num_responses, smoothing_function=SmoothingFunction().method7,
                                        exclude_indices={self._pad_index, self._end_index, self._start_index},
                                        prefix='_S_BLEU4')
        self.n_bleu2 = NLTKSentenceBLEU(ngram_weights=(1/2, 1/2),
                                        n_hyps=self._num_responses,
                                        exclude_indices={self._pad_index, self._end_index, self._start_index},
                                        prefix='_BLEU2')

        # We need our optimizer to know which parameters came from
        # which model, so we cheat by adding tags to them.
        for param in generator.parameters():
            setattr(param, '_generator', True)
        for param in discriminator.parameters():
            setattr(param, '_discriminator', True)

        self.generator = generator
        self.discriminator = discriminator
        self._disc_metrics = {
            "dfl": Average(),
            "dfacc": Average(),
            "drl": Average(),
            "dracc": Average(),
        }

        self._gen_metrics = {
            "_gl": Average(),
            "gce": Average(),
            "_mean": Average(),
            "_stdev": Average()
        }

    def encode_dialog(self, encoder: VariationalEncoder,
                      source_tokens: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.Tensor], temperature):
        query_dict = encoder(source_tokens)
        response_dict = encoder(target_tokens)
        query_dict = {'query_' + key: value for key, value in query_dict.items()}
        response_dict = {'response_' + key: value for key, value in response_dict.items()}
        dialog_dict = {**query_dict, **response_dict}
        query_latent = encoder.reparametrize(dialog_dict['query_prior'], dialog_dict['query_posterior'], temperature)
        response_latent = encoder.reparametrize(dialog_dict['response_prior'], dialog_dict['response_posterior'],
                                                temperature)
        if self.training:
            query_latent.requires_grad_()
            response_latent.requires_grad_()
        dialog_latent = torch.cat((query_latent, response_latent), dim=-1)
        dialog_dict['query_latent'] = query_latent
        dialog_dict['response_latent'] = response_latent
        dialog_dict['dialog_latent'] = dialog_latent
        dialog_dict['query'] = source_tokens
        dialog_dict['response'] = target_tokens
        return dialog_dict

    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor],  # Needed only for validation BLEU
                stage: List[str]):
        if self.training:
            stage = stage[0]
        else:
            stage = "generator"
        temperature = self.train_temperature if self.training else self.inference_temperature
        dialog_dict = self.encode_dialog(self._encoder, source_tokens, target_tokens, temperature)
        noised_query_latent = torch.cat([torch.randn_like(dialog_dict['query_latent']),
                                         dialog_dict['query_latent']], dim=1)
        if stage == "discriminator_real":
            dialog_latent = dialog_dict["dialog_latent"]
            batch_size = dialog_latent.size(0)
            device = dialog_latent.device
            labels = torch.ones([batch_size, 1]).to(device)
            output = self.discriminator(dialog_latent, labels)
            self._disc_metrics['drl'](output['loss'])
            self._disc_metrics['dracc'](output['accuracy'])
        elif stage == "discriminator_fake":
            predicted_latent = self.generator(noised_query_latent)["generation"]
            predicted_dialog = torch.cat([dialog_dict['response_latent'], dialog_dict['query_latent']])
            batch_size = predicted_latent.size(0)
            device = predicted_latent.device
            labels = torch.zeros([batch_size, 1]).to(device)
            output = self.discriminator(predicted_dialog, labels)
            self._disc_metrics['dfl'](output['loss'])
            self._disc_metrics['dfacc'](output['accuracy'])
        elif stage == "generator":
            output = self.generator(noised_query_latent, self.discriminator)
            predicted_response = output["generation"]
            self._gen_metrics['gce'](output['loss'])
            self._gen_metrics['_gl'](output['loss'])
            self._gen_metrics['_mean'](predicted_response.mean())
            self._gen_metrics['_stdev'](predicted_response.std())
            if not self.training:
                batch_size = predicted_response.size(0)
                responses = self.generator(predicted_response.repeat(self._num_responses, 1))["generation"]
                decoder_dict = self._decoder.generate(responses)

                # Be Careful with the permutation
                predictions = decoder_dict["predictions"].view(self._num_responses, batch_size, -1).permute(1, 0, 2)
                output.update({"predictions": predictions})
                if target_tokens:
                    self.s_bleu4(predictions, target_tokens["tokens"])
                    self.n_bleu2(predictions, target_tokens["tokens"])
        else:
            raise ValueError(f"Invalid stage: {stage}")
        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {key: float(metric.get_metric(reset=reset)) for key, metric in self._gen_metrics.items()}
        if self.training:
            metrics.update({key: float(metric.get_metric(reset=reset)) for key, metric in self._disc_metrics.items()})
        else:
            metrics.update(self.generator.get_metrics(reset=reset))
            metrics.update(self.s_bleu4.get_metric(reset=reset))
            metrics.update(self.n_bleu2.get_metric(reset=reset))
        return metrics

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
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_sentences = []
        for batch_indices in predicted_indices:
            # Check if multiple responses are generated for each sentence
            # if yes, decode all of them
            if len(batch_indices.shape) > 1:
                index_list = batch_indices.tolist()
            else:
                index_list = list(batch_indices.tolist())

            row_predicted_sentence = []
            for indices in index_list:
                # Collect indices till the first end_symbol
                if self._end_index in indices:
                    indices = indices[:indices.index(self._end_index)]
                predicted_tokens = [self.vocab.get_token_from_index(x) for x in indices]
                row_predicted_sentence.append(' '.join(predicted_tokens[1:]))
            all_predicted_sentences.append(row_predicted_sentence)
        output_dict["predicted_sentences"] = all_predicted_sentences
        return output_dict


@Model.register("generator")
class Generator(Model):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 activation: Activation = LeakyReLU(0.2),
                 initializer: InitializerApplicator = None):
        super().__init__(None)
        self._noise_mapper = Sequential(
            Linear(input_dim, 2*hidden_dim), BatchNorm1d(2*hidden_dim), activation,
            Linear(2*hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), activation,
            Linear(hidden_dim, output_dim)
        )
        self._ce_loss = BCEWithLogitsLoss()
        self._accuracy = BooleanAccuracy()
        initializer(self)

    @overrides
    def forward(self,
                noise_input: torch.Tensor,
                discriminator: Optional[Model] = None) -> Dict[str, torch.Tensor]:
        output = self._noise_mapper(noise_input)
        output_dict = {'generation': output}
        if discriminator is not None:
            predicted = discriminator(output)["output"]
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


@Model.register('discriminator')
class Discriminator(Model):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: Activation = LeakyReLU(0.2),
                 initializer: InitializerApplicator = None):
        super().__init__(None)
        self._activation = activation
        self._classifier = Sequential(
            Linear(input_dim, 2*hidden_dim), BatchNorm1d(2*hidden_dim), activation,
            Linear(2*hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), activation,
            Linear(hidden_dim, 1)
        )
        self._loss = BCEWithLogitsLoss()
        initializer(self)

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
