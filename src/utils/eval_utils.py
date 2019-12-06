#!/usr/bin/env python3

import logging
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


logger = logging.getLogger(__name__)


def compute_smoothed_bleu(dialog_dict, ngram=4):
    weights = tuple([1/ngram]*ngram)
    targets = [target.split() for target in dialog_dict['target']]
    num_responses = len(dialog_dict) - 2

    num_samples = len(dialog_dict['query'])
    bleu_scores = np.zeros((num_samples, num_responses))
    for rno in range(num_responses):
        responses = [response.split() for response in dialog_dict[f'response_{rno+1}']]
        assert len(targets) == len(responses)
        for sid, (ref, hyp) in enumerate(zip(targets, responses)):
            try:
                bleu_scores[sid][rno] = sentence_bleu(
                    [ref], hyp, smoothing_function=SmoothingFunction().method7,
                    weights=weights)
            except ZeroDivisionError:
                pass

    sbp = bleu_scores.mean()
    sbr = bleu_scores.max(axis=1).mean()
    sbf = 2*sbp*sbr/(sbp + sbr)
    bdict = {f'SBLEU{ngram}-P': sbp, f'SBLEU{ngram}-R': sbr, f'SBLEU{ngram}-F': sbf}
    return bdict


def batch_diversity(batch, n):
    """
    micro_diversity: (listof Str) -> Float
    Returns diversity of a given sentence
    """
    all_ngrams = []
    for sentence in batch:
        all_ngrams += list(nltk.ngrams(sentence.split(), n))
    if len(all_ngrams) == 0:
        diversity = np.nan
    else:
        diversity = len(set(all_ngrams)) / len(all_ngrams)
    return diversity


def intra_diversity(sentences, n):
    """
    intra_diversity: (listof (listof Str)) Int -> Float
    """
    intra_diversity = []
    for samples in sentences:
        sample_div = [batch_diversity([s], n) for s in samples]
        intra_diversity.append(np.nanmean(sample_div))
    return np.nanmean(intra_diversity)


def inter_diversity(sentences, n):
    """
    inter_diversity: (listof (listof Str)) Int -> Float
    """
    return np.nanmean([batch_diversity(samples, n) for samples in sentences])


def compute_diversity(dialog_dict):
    num_responses = len(dialog_dict) - 2
    responses_list = []
    for rno in range(num_responses):
        responses = [response for response in dialog_dict[f'response_{rno+1}']]
        responses_list.append(responses)
    flattened_responses = [response for responses in responses_list for response in responses]
    response_lens = [len(response.split()) for response in flattened_responses]
    diversity = {
        'intra1': intra_diversity(responses_list, 1),
        'intra2': intra_diversity(responses_list, 2),
        'inter1': inter_diversity(responses_list, 1),
        'inter2': inter_diversity(responses_list, 2),
        'asl': sum(response_lens)/len(response_lens)
    }
    return diversity
