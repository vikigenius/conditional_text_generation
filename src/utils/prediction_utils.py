#!/usr/bin/env python3

import os
import allennlp.nn.util as nn_util
from tqdm import tqdm
from allennlp.models import load_archive
from allennlp.data import Instance
from allennlp.data.iterators import BasicIterator
from typing import List, Dict
from src.models import Model


def get_responses(model: Model, device: int, instances: List[Instance],
                  num_responses: int, temperature: float = 1e-5,
                  flow: bool = True) -> List[List[str]]:
    iterator = BasicIterator(batch_size=128)
    iterator.index_with(model.vocab)
    predictions_dict = {f'response_{rno+1}': [] for rno in range(num_responses)}
    for batch in tqdm(iterator(instances, shuffle=False, num_epochs=1), desc='Predicting Responses'):
        for rno in range(num_responses):
            z = model.encode_query(nn_util.move_to_device(batch['source_tokens'], device),
                                   temperature=temperature)
            preds = model.decode_predictions(model._decoder(z)['predictions'])
            predictions_dict[f'response_{rno+1}'].extend(preds)
    return predictions_dict


def save_dialog_dict(dialog_dict: Dict[str, List[str]], save_path: str):
    ddkeys = list(dialog_dict.keys())
    ddvals = list(dialog_dict.values())
    num_samples = len(ddvals[0])

    with open(save_path, 'w') as sp:
        for sno in range(num_samples):
            for kno in range(len(ddkeys)):
                sp.write(ddkeys[kno] + ': ' + ddvals[kno][sno] + '\n')

            sp.write('\n')


def load_model(model_dir, epoch, cuda_device):
    weights_file = None
    if epoch > 0:
        weights_file = os.path.join(model_dir, f'model_state_epoch_{epoch}.th')
    archive_file = os.path.join(model_dir, 'model.tar.gz')
    model = load_archive(archive_file, cuda_device, weights_file=weights_file).model

    assert isinstance(model, Model)
    model.eval()
    return model
