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
    WordEmbeddings('de'),
    CharLMEmbeddings('german-forward'),
    CharLMEmbeddings('german-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter

search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256, 512])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 24, 32])

from pathlib import Path
from flair.hyperparameter.param_selection import SequenceTaggerParamSelector, OptimizationValue

param_selector = SequenceTaggerParamSelector(
    corpus,
    tag_type,
    Path('resources/results'),
    max_epochs=150,
    training_runs=3,
    optimization_value=OptimizationValue.DEV_SCORE
)

param_selector.optimize(search_space, max_evals=100)
