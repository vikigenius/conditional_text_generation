#!/usr/bin/env python3
import click
import logging
import pickle
from tqdm import tqdm
from allennlp.training.metrics import Average
from src.utils import file_utils, eval_utils


logger = logging.getLogger(__name__)


@click.group()
def evaluate():
    pass


@click.argument('lm_path', type=click.Path(exists=True))
@click.argument('pickle_path', type=click.Path(exists=True))
@evaluate.command()
def perplexity(lm_path, pickle_path):
    import kenlm
    model = kenlm.LanguageModel(lm_path)
    ppl = Average()
    with open(pickle_path, 'rb') as sf:
        dialog_dict = pickle.load(sf)
        num_responses = len(dialog_dict) - 2
        responses_list = []
        for rno in range(num_responses):
            responses = [response for response in dialog_dict[f'response_{rno+1}']]
            responses_list.append(responses)
        flattened_responses = [response for responses in responses_list for response in responses]
        with tqdm(flattened_responses, desc='Computing PPL') as pbar:
            for sentence in pbar:
                ppl(model.perplexity(sentence))
                pbar.set_postfix({'PPL': ppl.get_metric()})

    logger.info(f'PPL for file {pickle_path} = {ppl.get_metric()}')


@click.argument('pickle_path', type=click.Path(exists=True))
@click.option('--ngram', '-n', default=4)
@evaluate.command()
def overlap(pickle_path, ngram):
    with open(pickle_path, 'rb') as dfile:
        dialog_dict = pickle.load(dfile)
        print(eval_utils.compute_smoothed_bleu(dialog_dict, ngram=ngram))


@click.argument('pickle_path', type=click.Path(exists=True))
@evaluate.command()
def diversity(pickle_path):
    with open(pickle_path, 'rb') as dfile:
        dialog_dict = pickle.load(dfile)
        print(eval_utils.compute_diversity(dialog_dict))
