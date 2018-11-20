import gensim
import re

from flair.data import Sentence, TaggedCorpus, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List

columns = {0: 'text', 1: 'pos', 2: 'ner'}

corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_column_corpus(".", columns, train_file="ned.train",
                                                        dev_file="ned.testa",
                                                        test_file="ned.testb",
                                                        tag_to_biloes='ner')

tag_type = 'ner'



tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

print(corpus)

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('wiki.nl.vec', binary=False)
word_vectors.save('wiki.nl.vec.gensim')

custom_embedding = WordEmbeddings('wiki.nl.vec.gensim')

char_lm_forward = CharLMEmbeddings('lm-nl-large-forward-v0.1.pt')
char_lm_backward = CharLMEmbeddings('lm-nl-large-backward-v0.1.pt')

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

trainer.train('resources/taggers/ner-dutch',
              learning_rate=0.1,
              mini_batch_size=8,
              max_epochs=500)
