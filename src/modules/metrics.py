#!/usr/bin/env python

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric
from typing import Optional, Iterable, Callable, Set, Dict
import math
import numpy as np
from allennlp.training.metrics.entropy import Entropy
from nltk.translate.bleu_score import sentence_bleu


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
    def __init__(self,
                 ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
                 smoothing_function: Optional[Callable] = None,
                 n_refs: int = 1,
                 n_hyps: int = 1,
                 exclude_indices: Set[int] = None,
                 auto_reweigh: bool = False) -> None:
        super().__init__()
        self.sentence_bleu = lambda ref, hyp: sentence_bleu(
            ref, hyp, ngram_weights, smoothing_function=smoothing_function, auto_reweigh=auto_reweigh)

        self._exclude_indices = exclude_indices or set()
        self.n_refs = n_refs
        self.n_hyps = n_hyps
        self.scores = np.zeros([n_hyps, n_refs])
        self.count = 0

    @overrides
    def reset(self):
        self.scores = np.zeros([self.n_hyps, self.n_refs])
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
        for hno in range(self.n_hyps):
            for rno in range(self.n_refs):
                batch_bleu = 0.0
                valid_batch_count = batch_size
                for batchno in range(batch_size):
                    reference = gold_targets[batchno, rno, :].tolist()
                    hypothesis = predictions[batchno, hno, :].tolist()
                    hypothesis_tokens = [htoken for htoken in hypothesis if htoken not in self._exclude_indices]
                    reference_tokens = [rtoken for rtoken in reference if rtoken not in self._exclude_indices]
                    try:
                        batch_bleu += self.sentence_bleu([reference_tokens], hypothesis_tokens)
                    except ZeroDivisionError:
                        valid_batch_count -= 1
                if valid_batch_count == 0:
                    raise ValueError("All hypothesis for hno: {hno} in batch has length zero")
                self.scores[hno][rno] += batch_bleu/valid_batch_count

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        precision_bleu = self.scores.max(1).mean()/self.count if self.count else 0
        recall_bleu = self.scores.max(0).mean()/self.count if self.count else 0
        f_bleu = 2*precision_bleu*recall_bleu/(precision_bleu + recall_bleu + 1e-13)
        if reset:
            self.reset()
        return {"_P-BLEU": precision_bleu, '_R-BLEU': recall_bleu, 'F-BLEU': f_bleu}
