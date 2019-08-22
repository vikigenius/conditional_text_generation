#!/usr/bin/env python

from typing import Dict, Tuple
import torch
from torch.nn.modules import Dropout, Linear
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from src.modules.decoders.decoder import Decoder


DecoderState = Tuple[torch.Tensor, torch.Tensor]


@Decoder.register("variational_decoder")
class VariationalDecoder(Decoder):
    """
    This ``VariationalDecoder`` Trains a Variational Decoder given the latent variable for
    the Language Modeling task
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    target_embedder : ``TextFieldEmbedder``, required
        Embedder for target side sequences
    rnn : ``Seq2SeqEncoder``, required
        The decoder of the "encoder/decoder" model
    latent_dim : ``int``, required
        The dimention of the latent, z vector. This is not necessarily the same size as the encoder
        output dim
    dropout_p : ``float``, optional (default = 0.5)
        This scalar is used to twice. once as a scalar for the dropout to the input embeddings for the
        decoder. Imitating word-dropout, and once as a dropout layer after the decoder RNN.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 target_embedder: TextFieldEmbedder,
                 rnn: Seq2SeqEncoder,
                 latent_dim: int,
                 dropout_p: float = 0.5) -> None:
        super(VariationalDecoder, self).__init__()
        self.vocab = vocab
        self._target_embedder = target_embedder
        self.rnn = rnn
        self.dec_num_layers = self.rnn._module.num_layers  # pylint: disable=protected-access
        self.dec_hidden = self.rnn._module.hidden_size  # pylint: disable=protected-access

        self._latent_to_dec_hidden = Linear(latent_dim, self.dec_hidden)
        self._dec_linear = Linear(self.rnn.get_output_dim(), self.vocab.get_vocab_size())

        self._emb_dropout = Dropout(dropout_p)
        self._dec_dropout = Dropout(dropout_p)

        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)

    def forward(self,  # type: ignore
                latent: torch.Tensor,
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make a forward pass of the decoder, given the latent vector and the sent as references.
        Notice explanation in simple_seq2seq.py for creating relavant targets and mask
        """
        if self.training and target_tokens is not None:
            target_mask = get_text_field_mask(target_tokens)
            relevant_targets = {"tokens": target_tokens["tokens"][:, :-1]}
            relevant_mask = target_mask[:, :-1]
            embeddings, state = self._prepare_decoder(latent, relevant_targets)
            logits = self._run_decoder(embeddings, relevant_mask, state, latent)

            class_probabilities = F.softmax(logits, 2)
            _, best_predictions = torch.max(class_probabilities, 2)

            loss = self._get_reconstruction_loss(logits, target_tokens['tokens'], target_mask)
            output_dict = {'logits': logits, 'predictions': best_predictions, 'loss': loss}
        else:
            if target_tokens is not None:
                output_dict = self.generate(latent, target_tokens, max_len=target_tokens['tokens'].size(1))
            else:
                output_dict = self.generate(latent, target_tokens)
        return output_dict

    def _prepare_decoder(self, latent: torch.Tensor,
                         relevant_targets: torch.Tensor) -> Tuple[torch.Tensor, DecoderState]:
        embeddings = self._target_embedder(relevant_targets)
        embeddings = self._emb_dropout(embeddings)
        h0 = embeddings.new_zeros(
            self.dec_num_layers, embeddings.size(0), self.dec_hidden)  # pylint: disable=invalid-name
        c0 = embeddings.new_zeros(
            self.dec_num_layers, embeddings.size(0), self.dec_hidden)  # pylint: disable=invalid-name
        h0[-1] = self._latent_to_dec_hidden(latent)
        state = (h0, c0)
        return embeddings, state

    def _run_decoder(self,
                     embeddings: torch.Tensor, relevant_mask: torch.LongTensor,
                     state: DecoderState, latent: torch.Tensor) -> torch.Tensor:
        expanded_latent = latent.unsqueeze(1).expand(embeddings.size(0), embeddings.size(1), latent.size(1))
        dec_input = torch.cat([embeddings, expanded_latent], 2)
        decoder_out = self.rnn(dec_input, relevant_mask, state)
        decoder_out = self._dec_dropout(decoder_out.contiguous().view(embeddings.size(0)*embeddings.size(1), -1))
        logits = self._dec_linear(decoder_out)
        logits = logits.view(embeddings.size(0), embeddings.size(1), -1)

        return logits

    def _get_reconstruction_loss(self, logits: torch.Tensor,
                                 targets: torch.Tensor,
                                 target_mask: torch.Tensor):
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()
        reconstruction_loss_per_token = sequence_cross_entropy_with_logits(logits, relevant_targets,
                                                                           relevant_mask, average=None)
        # In VAE, we want both loss terms to be comparable, for this reason, we compare kld to each
        # predicted token NLL. For this reason we return to the following mean.
        reconstruction_loss = (reconstruction_loss_per_token * (relevant_mask.sum(1).float() + 1e-13)).mean()
        return reconstruction_loss

    def generate(self, latent: torch.Tensor, target_tokens: Dict[str, torch.LongTensor] = None,
                 reset_states=True, max_len: int = 30) -> Dict[str, torch.Tensor]:
        batch_size, _ = latent.size()
        if reset_states:
            self.rnn.reset_states()
        last_genereation = latent.new_full((batch_size, 1),
                                           fill_value=self._start_index, dtype=torch.long)
        h0 = latent.new_zeros(self.dec_num_layers, batch_size, self.dec_hidden)  # pylint: disable=invalid-name
        c0 = latent.new_zeros(self.dec_num_layers, batch_size, self.dec_hidden)  # pylint: disable=invalid-name
        h0[-1] = self._latent_to_dec_hidden(latent)
        # We are decoding step by step. So we are using a stateful decoder
        self.rnn.stateful = True
        restoration_indices = torch.arange(batch_size)
        restoration_indices = restoration_indices.to(h0.device)
        self.rnn._update_states((h0, c0), restoration_indices)  # pylint: disable=protected-access
        generations = [last_genereation]
        all_logits = []
        for _ in range(max_len):
            embeddings = self._target_embedder({"tokens": last_genereation})
            mask = get_text_field_mask({"tokens": last_genereation})
            logits = self._run_decoder(embeddings, mask, None, latent)
            class_probabilities = F.softmax(logits, 2)
            _, last_genereation = torch.max(class_probabilities, 2)
            generations.append(last_genereation)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits[:-1], 1)
        generations = torch.cat(generations, 1)
        self.rnn.stateful = False

        output_dict = {"logits": all_logits, "predictions": generations}
        if target_tokens is not None:
            target_mask = get_text_field_mask(target_tokens)
            output_dict['loss'] = self._get_reconstruction_loss(all_logits, target_tokens['tokens'], target_mask)
        return output_dict
