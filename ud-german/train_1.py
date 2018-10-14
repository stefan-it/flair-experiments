from flair.data import Sentence, TaggedCorpus, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List

sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_German/de-ud-train.conllu")
sentences_dev: List[Sentence]   = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_German/de-ud-dev.conllu")
sentences_test: List[Sentence]  = NLPTaskDataFetcher.read_conll_ud("universal-dependencies-1.2/UD_German/de-ud-test.conllu")

corpus: TaggedCorpus = TaggedCorpus(sentences_train, sentences_dev,
                                    sentences_test)

tag_type = 'upos'

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

trainer.train('resources/taggers/ud-german',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=500)
