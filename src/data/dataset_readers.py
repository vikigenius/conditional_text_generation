#!/usr/bin/env python

import csv
import logging
from overrides import overrides
from typing import Dict
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField, LabelField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("autoencoder")
class AutoencoderDatasetReader(DatasetReader):
    """
    ``AutoencoderDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 delimiter: str = "\t",
                 max_seq_len: int = 30,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._delimiter = delimiter
        self._max_seq_len = max_seq_len

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for _, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self, input_string: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._source_tokenizer.tokenize(input_string)
        tokenized_string = tokenized_string[:self._max_seq_len - 2]
        tokenized_source = tokenized_string.copy()
        tokenized_target = tokenized_string.copy()
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)
        return Instance({"source_tokens": source_field, "target_tokens": target_field})


@DatasetReader.register("dialog-classification")
class DialogClassifierDatasetReader(Seq2SeqDatasetReader):
    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if len(row) != 3:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (row, line_num + 1))

                query, response, label = row
                yield self.text_to_instance(query, response, int(label))

    @overrides
    def text_to_instance(self, query_string: str, response_string: str, label: int) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._source_tokenizer.tokenize(query_string)
        tokenized_source = tokenized_string.copy()
        tokenized_target = tokenized_string.copy()
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        tokenized_target = self._target_tokenizer.tokenize(response_string)
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)

        label_field = LabelField(label, skip_indexing=True)
        return Instance({"query_tokens": source_field, "response_tokens": target_field, "label": label_field})


@DatasetReader.register('dialog')
class DialogDatasetReader(Seq2SeqDatasetReader):
    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if len(row) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (row, line_num + 1))
                source_sequence, target_sequence = row
                yield self.text_to_instance(source_sequence, target_sequence)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})
