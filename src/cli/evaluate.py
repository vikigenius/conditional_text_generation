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
@click.argument('sample_file', type=click.Path(exists=True))
@evaluate.command()
def perplexity(lm_path, sample_file):
    import kenlm
    model = kenlm.LanguageModel(lm_path)
    ppl = Average()
    num_lines = file_utils.get_num_lines(sample_file)
    with open(sample_file) as sf:
        with tqdm(sf, total=num_lines, desc='Computing PPL') as pbar:
            for sentence in pbar:
                ppl(model.perplexity(sentence))
                pbar.set_postfix({'PPL': ppl.get_metric()})

    logger.info(f'PPL for file {sample_file} = {ppl.get_metric()}')


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
