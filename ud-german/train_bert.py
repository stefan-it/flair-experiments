import sys

from flair.data import Sentence, TaggedCorpus, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings
from typing import List

layers = sys.argv[1]

sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_German/de-ud-train.conllu")
sentences_dev: List[Sentence]   = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_German/de-ud-dev.conllu")
sentences_test: List[Sentence]  = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_German/de-ud-test.conllu")

corpus: TaggedCorpus = TaggedCorpus(sentences_train, sentences_dev,
                                    sentences_test)

tag_type = 'upos'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

custom_embedding = WordEmbeddings('eu')

embedding_types: List[TokenEmbeddings] = [
    custom_embedding,
    BertEmbeddings('bert-base-multilingual-cased', layers=f'{layers}')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=512,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False)

from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train(f'resources/taggers/ud-basque-bert-fasttext-layer_{layers}',
              EvaluationMetric.MICRO_ACCURACY,
              learning_rate=0.1,
              mini_batch_size=8,
              max_epochs=500)
