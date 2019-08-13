"""
Borrowed from https://github.com/mhagiwara/realworldnlp/blob/master/examples/mt/mt.py
"""

import itertools

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from src.modules.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer

EN_EMBEDDING_DIM = 300
ZH_EMBEDDING_DIM = 300
HIDDEN_DIM = 512
CUDA_DEVICE = 0


def main():
    reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=WordTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read('data/interim/dialog/train_dialog.tsv')
    validation_dataset = reader.read('data/interim/dialog/valid_dialog.tsv')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, num_layers=2, batch_first=True, bidirectional=True))

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()
    # attention = None

    max_decoding_steps = 30
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=3,
                          use_bleu=True).to('cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=1,
                      grad_norm=5,
                      grad_clipping=5,
                      cuda_device=CUDA_DEVICE)

    for i in range(50):
        print('Epoch: {}'.format(i))
        trainer.train()

        predictor = SimpleSeq2SeqPredictor(model, reader)
        all_sents = []
        for instance in itertools.islice(validation_dataset, len(validation_dataset)):
            sent = ' '.join(predictor.predict_instance(instance)['predicted_tokens'])
            all_sents.append(sent)
        with open('reports/outputs/s2s/generated_sentences_{}.txt'.format(i), 'w') as f:
            f.write('\n'.join(all_sents))


if __name__ == '__main__':
    main()
