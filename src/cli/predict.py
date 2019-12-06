#!/usr/bin/env python3

import torch
import os
import click
import logging
import pickle
from src.data.dataset import SentenceReader
from src.utils import prediction_utils


logger = logging.getLogger(__name__)


@click.group()
def predict():
    logging.getLogger('allennlp').setLevel(logging.ERROR)
    torch.set_grad_enabled(False)


@click.command()
@click.pass_obj
def samples(model):
    raise NotImplementedError


@click.argument('query_file', type=click.Path(exists=True))
@click.option('--model_dir', default='models/dialog', type=click.Path(exists=True))
@click.option('--epoch', default=-1)
@click.option('--cuda_device', default=0)
@click.option('--target_prefix', type=str, default='dialog')
@click.option('--target_file', type=click.Path(exists=True))
@click.option('--temperature', default=1.0)
@click.option('--num_responses', '-n', default=10, help='Number of responses for each query')
@predict.command()
def dialog(query_file, model_dir, epoch, cuda_device, target_prefix, target_file,
           temperature, num_responses):
    reader = SentenceReader()
    source_instances = reader.read(query_file)
    model = prediction_utils.load_model(model_dir, epoch, cuda_device)
    pred_dict = prediction_utils.get_responses(model, cuda_device, source_instances,
                                               num_responses, temperature=temperature)

    # Add the targets and source
    pred_dict.update({'query': []})
    for instance in source_instances:
        pred_dict['query'].append(' '.join([str(token) for token in instance['source_tokens'][1:-1]]))
    if target_file:
        pred_dict.update({'target': []})
        reader = SentenceReader(key='target_tokens')
        target_instances = reader.read(target_file)
        for instance in target_instances:
            pred_dict['target'].append(' '.join([str(token) for token in instance['target_tokens'][1:-1]]))
    pickle_path = os.path.join('data/interim/dialog/', target_prefix + '.pkl')
    logger.info(f'Dumping Pickle to {pickle_path}')
    with open(pickle_path, 'wb') as pkl:
        pickle.dump(pred_dict, pkl)

    dialog_path = os.path.join('data/outputs/dialog/', target_prefix + '.txt')
    logger.info(f'Saving dialog to {dialog_path}')
    prediction_utils.save_dialog_dict(pred_dict, dialog_path)
