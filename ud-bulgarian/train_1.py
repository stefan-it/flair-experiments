import gensim
import re

from flair.data import Sentence, TaggedCorpus, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List

sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_Bulgarian/bg-ud-train.conllu")
sentences_dev: List[Sentence]   = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_Bulgarian/bg-ud-dev.conllu")
sentences_test: List[Sentence]  = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_Bulgarian/bg-ud-test.conllu")

corpus: TaggedCorpus = TaggedCorpus(sentences_train, sentences_dev,
                                    sentences_test)

tag_type = 'upos'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('wiki.bg.vec', binary=False)
word_vectors.save('wiki.bg.vec.gensim')

custom_embedding = WordEmbeddings('custom', 'wiki.bg.vec.gensim')

char_lm_forward = CharLMEmbeddings('lm-bg-small-forward-v0.1.pt')
char_lm_backward = CharLMEmbeddings('lm-bg-small-backward-v0.1.pt')

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

from flair.trainers import SequenceTaggerTrainer

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus,
                                                       test_mode=True)

trainer.train('resources/taggers/ud-bulgarian',
              learning_rate=0.1,
              mini_batch_size=8,
              max_epochs=500)
