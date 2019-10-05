#!/usr/bin/env python
import numpy as np
import sys
from allennlp.common import Registrable


class LossWeight(Registrable):
    """
    This class is an abstract class for loss weighters.
    Whenever the loss function is composed of more then a single term weighting is probable.
    Use children of this class for different constant/annealed weights
    """
    def __init__(self, initial_weight: float) -> None:
        self._weight = initial_weight

    def next(self) -> float:
        self.step()
        return self.get()

    def get(self) -> float:
        return self._weight

    def step(self):
        raise NotImplementedError


@LossWeight.register("constant_weight")
class ConstantWeight(LossWeight):
    """
    This class is for a constant weight scalar.
    """
    def step(self) -> None:
        pass


class AnnealedWeight(LossWeight):
    def __init__(self, min_weight: float = 0.0, max_weight: float = 1.0,
                 warmup: int = 0, early_stop_iter: int = sys.maxsize) -> None:
        """
        This class is an abstract class for annealing loss weighters.
        """
        super().__init__(min_weight)
        self.iteration = 0
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup = warmup
        self.early_stop_iter = early_stop_iter

    def step(self):
        weight = self._get_weight() if self.iteration > self.warmup else 0.0
        if self.iteration < self.early_stop_iter:
            self._weight = weight
        self._weight = min(self._weight, self.max_weight)
        self._weight = max(self._weight, self.min_weight)
        self.iteration += 1

    def _get_weight(self):
        raise NotImplementedError


@LossWeight.register("linear_annealed")
class LinearAnnealedWeight(AnnealedWeight):
    """
    This class anneals weights linearly.
    """
    def __init__(self, slope: float, intercept: float, min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 warmup: int = 0, early_stop_iter: int = sys.maxsize) -> None:
        super().__init__(min_weight, max_weight, warmup, early_stop_iter)
        self.slope = slope
        self.intercept = intercept

    def _get_weight(self):
        return self.slope*self.iteration + self.intercept


@LossWeight.register("tanh_annealed")
class TanhAnnealedWeight(AnnealedWeight):
    """
    This class anneals weights in a tanh fashion
    """
    def __init__(self, slope: float, margin: float, min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 warmup: int = 0, early_stop_iter: int = sys.maxsize) -> None:
        super().__init__(min_weight, max_weight, warmup, early_stop_iter)
        self.slope = slope
        self.margin = margin

    def _get_weight(self):
        return 0.5*(np.tanh(self.slope*(self.iteration - self.margin)) + 1)


@LossWeight.register("sigmoid_annealed")
class SigmoidAnnealedWeight(AnnealedWeight):
    """
    This class anneals weights in a sigmoid.
    """
    def __init__(self, slope: float, margin: float, min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 warmup: int = 0, early_stop_iter: int = sys.maxsize) -> None:
        super().__init__(min_weight, max_weight, warmup, early_stop_iter)
        self.slope = slope
        self.margin = margin

    def _get_weight(self):
        return 1/(1 + np.exp(-self.slope*(self.iteration - self.margin)))
