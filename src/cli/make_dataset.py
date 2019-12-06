#!/usr/bin/env python3

import click
import logging
from tqdm import tqdm
from src.utils import file_utils


logger = logging.getLogger(__name__)


@click.group()
def make_dataset():
    pass


@click.argument('output_path', type=click.Path())
@click.argument('input_path', type=click.Path(exists=True))
@make_dataset.command()
def dwae_gen_lm(input_path, output_path):
    num_lines = file_utils.get_num_lines(input_path)
    with open(input_path) as ip, open(output_path, 'w') as op:
        with tqdm(ip, total=num_lines, desc='Computing PPL') as pbar:
            for line in pbar:
                if line.strip().startswith('Sample'):
                    op.write(line.strip()[12:-4] + '\n')
