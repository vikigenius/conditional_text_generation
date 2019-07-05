#!/usr/bin/env python

import torch
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
    def __init__(self, min_weight: float, max_weight: float, warmup: int, num_iter_to_max: int) -> None:
        """
        This class is an abstract class for annealing loss weighters.
        """
        super().__init__(min_weight)
        self.iteration = 0
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup = warmup
        self.num_iter_to_max = num_iter_to_max

    def step(self):
        self._take_step()
        self.iteration += 1

    def _take_step(self):
        raise NotImplementedError


@LossWeight.register("linear_annealed")
class LinearAnnealedWeight(AnnealedWeight):
    """
    This class anneals weights linearly.
    """
    def _take_step(self):
        if self.iteration < self.warmup:
            self._weight = self.min_weight
        elif self.num_iter_to_max is not None and self.iteration > self.num_iter_to_max:
            self._weight = self.max_weight
        else:
            offset = (self.max_weight-self.min_weight)*(self.iteration-self.warmup)/(self.num_iter_to_max-self.warmup)
            self._weight = self.min_weight + offset


@LossWeight.register("sigmoid_annealed")
class SigmoidAnnealedWeight(AnnealedWeight):
    """
    This class anneals weights in a sigmoid fashion.
    """
    def __init__(self, min_weight: float,
                 max_weight: float,
                 warmup: int,
                 num_iter_to_max: int,
                 slope: float) -> None:
        super().__init__(min_weight, max_weight, warmup, num_iter_to_max)
        self.slope = slope

    def _take_step(self):
        shifted_max = self.max_weight - self.min_weight
        middle_point = (self.warmup + self.num_iter_to_max)/2
        res = shifted_max * torch.sigmoid(torch.Tensor([self.slope*(self.iteration-middle_point)])) + self.min_weight
        self._weight = round(res.item(), 2)
