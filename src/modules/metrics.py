#!/usr/bin/env python
import math
import torch
import logging
import numpy as np
from overrides import overrides
from allennlp.training.metrics.metric import Metric
from typing import Optional, Iterable, Callable, Set, Dict, Union
from allennlp.training.metrics.entropy import Entropy
from nltk.translate.bleu_score import sentence_bleu


logger = logging.getLogger(__name__)


@Metric.register("lm-perplexity")
class LMPerplexity(Metric):
    def __init__(self) -> None:
        super().__init__()
        self._entropy = Entropy()

    @overrides
    def __call__(self,  # type: ignore
                 logits: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        self._entropy(logits, mask.float())

    @overrides
    def get_metric(self, reset: bool = False):
        entropy_val = self._entropy.get_metric(reset)
        ppl = math.exp(min(entropy_val, 100))
        return {"PPL": ppl}

    @overrides
    def reset(self):
        self._entropy.reset()


@Metric.register("nltk_sentence_bleu")
class NLTKSentenceBLEU(Metric):
    """
    Bilingual Evaluation Understudy (BLEU).
    BLEU is a common metric used for evaluating the quality of machine translations
    against a set of reference translations. See `Papineni et. al.,
    "BLEU: a method for automatic evaluation of machine translation", 2002
    <https://www.semanticscholar.org/paper/8ff93cfd37dced279134c9d642337a2085b31f59/>`_.
    Parameters
    ----------
    ngram_weights : ``Iterable[float]``, optional (default = (0.25, 0.25, 0.25, 0.25))
        Weights to assign to scores for each ngram size.
    smoothing_function : ``Callable``, optional (default = None)
        The smoothing function to use.
    n_refs: ``int``, optional (default = 1)
    n_hyps: ``int``, optional (default = 1)
    exclude_indices : ``Set[int]``, optional (default = None)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.
    auto_reweigh: ``bool``, optional (default = False)
    """
    def __init__(self,
                 ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
                 smoothing_function: Optional[Callable] = None,
                 n_refs: int = 1,
                 n_hyps: int = 1,
                 exclude_indices: Set[int] = None,
                 auto_reweigh: bool = False,
                 prefix: Union[str, Iterable[str]] = 'BLEU') -> None:
        super().__init__()
        self.sentence_bleu = lambda ref, hyp: sentence_bleu(
            ref, hyp, ngram_weights, smoothing_function=smoothing_function, auto_reweigh=auto_reweigh)

        self._prefix = [prefix]*3 if isinstance(prefix, str) else prefix

        if len(self._prefix) != 3:
            raise ValueError('Expected 3 prefixes got {len(self._prefix)} instead')

        self._exclude_indices = exclude_indices or set()
        self.n_refs = n_refs
        self.n_hyps = n_hyps
        self.precision_bleus = []
        self.recall_bleus = []
        self.count = 0

    @overrides
    def reset(self):
        self.precision_bleus = []
        self.recall_bleus = []
        self.count = 0

    @overrides
    def __call__(self,  # type: ignore
                 predictions: torch.LongTensor,
                 gold_targets: torch.LongTensor) -> None:
        """
        Update precision counts.

        Parameters
        ----------
        predictions : ``torch.LongTensor``, required
            Batched predicted tokens of shape `(batch_size, n_hyps, max_sequence_length)`.
        references : ``torch.LongTensor``, required
            Batched reference (gold) translations with shape `(batch_size, n_refs, max_gold_sequence_length)`.

        Returns
        -------
        None
        """
        self.count += 1
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(1)
        if gold_targets.dim() == 2:
            gold_targets = gold_targets.unsqueeze(1)
        predictions, gold_targets = self.unwrap_to_tensors(predictions, gold_targets)
        batch_size = predictions.size(0)
        bleu_scores = np.zeros([batch_size, self.n_hyps, self.n_refs])
        for bidx in range(batch_size):
            for hno in range(self.n_hyps):
                for rno in range(self.n_refs):
                    reference = gold_targets[bidx, rno, :].tolist()
                    hypothesis = predictions[bidx, hno, :].tolist()
                    hypothesis_tokens = [htoken for htoken in hypothesis if htoken not in self._exclude_indices]
                    reference_tokens = [rtoken for rtoken in reference if rtoken not in self._exclude_indices]
                    try:
                        bleu_scores[bidx][hno][rno] = self.sentence_bleu([reference_tokens], hypothesis_tokens)
                    except FloatingPointError:
                        logger.warn('Division by Zeror error occured, setting bleu to zero')
        recall_bleu = bleu_scores.max(1).mean()
        precision_bleu = bleu_scores.max(2).mean()
        self.recall_bleus.append(recall_bleu)
        self.precision_bleus.append(precision_bleu)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        precision_bleu = sum(self.precision_bleus)/self.count if self.count else 0
        recall_bleu = sum(self.recall_bleus)/self.count if self.count else 0
        f_bleu = 2*precision_bleu*recall_bleu/(precision_bleu + recall_bleu + 1e-13)
        if reset:
            self.reset()
        return {
            f"{self._prefix[0]}P": precision_bleu,
            f'{self._prefix[1]}R': recall_bleu,
            f'{self._prefix[2]}F': f_bleu
        }
