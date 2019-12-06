#!/usr/bin/env python

import csv
import logging
from overrides import overrides
from typing import Dict
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("autoencoder")
class AutoencoderDatasetReader(Seq2SeqDatasetReader):
    """
    ``AutoencoderDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
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


@DatasetReader.register('dialog-gan')
class DialogGanDatasetReader(Seq2SeqDatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.
    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    batch_size: int, (optional, default=32)
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 delimiter: str = "\t",
                 lazy: bool = False) -> None:
        super().__init__(source_tokenizer, target_tokenizer,
                         source_token_indexers, target_token_indexers,
                         source_add_start_token, delimiter, lazy)
        self.states = ["discriminator_real", "discriminator_fake", "generator"]
        self.state = self._state_generator()

    def _state_generator(self):
        while True:
            for state in self.states:
                yield state

    def _get_state(self):
        return next(self.state)

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
        stage_field = MetadataField(self._get_state())
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
            return Instance({"source_tokens": source_field, "target_tokens": target_field, "stage": stage_field})
        else:
            return Instance({'source_tokens': source_field})


@DatasetReader.register("sentence")
class SentenceReader(DatasetReader):
    """
    ``AutoencoderDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 delimiter: str = "\t",
                 max_seq_len: int = 30,
                 key: str = 'source_tokens',
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._add_start_token = add_start_token
        self._delimiter = delimiter
        self._max_seq_len = max_seq_len
        self._key = key

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for _, line in enumerate(data_file):
                sentence = line.strip("\n")
                if sentence:
                    yield self.text_to_instance(sentence)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(sentence)
        tokenized_string = tokenized_string[:self._max_seq_len - 2]
        if self._add_start_token:
            tokenized_string.insert(0, Token(START_SYMBOL))
        tokenized_string.append(Token(END_SYMBOL))
        sentence_field = TextField(tokenized_string, self._token_indexers)

        return Instance({self._key: sentence_field})
