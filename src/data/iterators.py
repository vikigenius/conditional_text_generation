#!/usr/bin/env python
import random
from collections import defaultdict
from overrides import overrides
from typing import List, Tuple, Iterable, Dict
from allennlp.data.iterators import DataIterator
from allennlp.data.iterators.bucket_iterator import sort_by_padding
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance


@DataIterator.register("homogeneous_bucket")
class HomogeneousBucketIterator(DataIterator):
    """
    An iterator that takes in heterogenuous instances and returns homogeneous batches determined by
    partition_key and by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).
    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.
        For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
        "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
        "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
        "num_token_characters" of the "sentence1" field.  TODO(mattg): we should have some
        documentation somewhere that gives the standard padding keys used by different fields.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.
        Note that if you specify ``max_instances_in_memory``, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    partition_key : ``str``, optional, (default = "dataset")
        The key of the ``MetadataField`` indicating what "type" of instance this is.
    """
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 partition_key: str = "dataset") -> None:
        if not sorting_keys:
            raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory)
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        self._partition_key = partition_key

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):

            # WARNING: Assumes that defaultdict is ordered i.e > python 3.6
            # Divvy up the instances based on their value of the "partition_key" field.
            hoppers: Dict[str, List[Instance]] = defaultdict(list)
            for instance in instance_list:
                partition = instance.fields[self._partition_key].metadata  # type: ignore
                hoppers[partition].append(instance)

            # Shuffle each parition separately
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                for k, v in hoppers.items():
                    random.shuffle(v)

            # Sort each partition spearately
            hoppers = {key: sort_by_padding(value, self._sorting_keys, self.vocab, self._padding_noise) for key, value in hoppers.items()}

            instance_list = sort_by_padding(instance_list,
                                            self._sorting_keys,
                                            self.vocab,
                                            self._padding_noise)
            # Get a `lazy_groups_of` iterator over each set of homogeneous instances.
            batches = {key: lazy_groups_of(iter(hopper), self._batch_size) for key, hopper in hoppers.items()}

            remaining = set(batches)
            # Yield batches in a round-robin fashion until none are left.

            keys = batches.keys()

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches[keys[0]].pop()
                penultimate_batch = batches[keys[0]].pop()
                batches[keys[0]].insert(0, penultimate_batch)
                batches[keys[0]].insert(0, last_batch)

            while remaining:
                for key, lazy_batches in batches.items():
                    if key in remaining:
                        try:
                            batch = next(lazy_batches)
                            yield Batch(batch)
                        except StopIteration:
                            remaining.remove(key)
