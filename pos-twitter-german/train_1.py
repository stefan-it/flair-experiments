import re

from flair.data import Sentence, TaggedCorpus, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List

def read_conll_2_column_data(path_to_conll_file: str, tag_name: str):
    sentences: List[Sentence] = []

    lines: List[str] = open(path_to_conll_file).read().strip().split('\n')

    sentence: Sentence = Sentence()
    for line in lines:
        if line == '':
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence: Sentence = Sentence()
        else:
            fields: List[str] = re.split("\s+", line)
            token = Token(fields[0])
            token.add_tag(tag_name, fields[1])
            sentence.add_token(token)

    if len(sentence.tokens) > 0:
        sentences.append(sentence)

    return sentences

sentences_train: List[Sentence] = read_conll_2_column_data("train.txt", "pos")
sentences_dev: List[Sentence]   = read_conll_2_column_data("dev.txt", "pos")
sentences_test: List[Sentence]  = read_conll_2_column_data("test.txt", "pos")

# Training, dev and test data set size from paper
number_train_tweets = 420
number_dev_tweets = 500
number_test_tweets = 506

# Check, if our dataset reader works
assert len(sentences_train) == number_train_tweets
assert len(sentences_dev) == number_dev_tweets
assert len(sentences_test) == number_test_tweets

corpus: TaggedCorpus = TaggedCorpus(sentences_train, sentences_dev,
                                    sentences_test)

tag_type = 'pos'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de-fasttext'),
    CharLMEmbeddings('german-forward'),
    CharLMEmbeddings('german-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

from flair.trainers import SequenceTaggerTrainer

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus,
                                                       test_mode=True)

trainer.train('resources/taggers/pos-twitter-german',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
