import gensim
import re

from flair.data import Sentence, TaggedCorpus, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List

sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_Basque/eu-ud-train.conllu")
sentences_dev: List[Sentence]   = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_Basque/eu-ud-dev.conllu")
sentences_test: List[Sentence]  = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_Basque/eu-ud-test.conllu")

corpus: TaggedCorpus = TaggedCorpus(sentences_train, sentences_dev,
                                    sentences_test)

tag_type = 'upos'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

#word_vectors = gensim.models.KeyedVectors.load_word2vec_format('wiki.eu.vec', binary=False)
#word_vectors.save('wiki.eu.vec.gensim')

custom_embedding = WordEmbeddings('wiki.eu.vec.gensim')

char_lm_forward = CharLMEmbeddings('lm-eu-large-forward-v0.1.pt')
char_lm_backward = CharLMEmbeddings('lm-eu-large-backward-v0.1.pt')

embedding_types: List[TokenEmbeddings] = [
    custom_embedding,
    char_lm_forward,
    char_lm_backward
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=512,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/ud-basque',
              EvaluationMetric.MICRO_ACCURACY,
              learning_rate=0.1,
              mini_batch_size=8,
              max_epochs=500)
