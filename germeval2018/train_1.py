from typing import List

from flair.data import Sentence, TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentLSTMEmbeddings
from flair.models.text_classification_model import TextClassifier
from flair.trainers.text_classification_trainer import TextClassifierTrainer

sentences_train: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file('train_flair.txt.preprocessed')
sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file('dev_flair.txt.preprocessed')
sentences_test: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file('test_flair.txt.preprocessed')

corpus = TaggedCorpus(sentences_train, sentences_dev, sentences_test)

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [WordEmbeddings('de-fasttext'),
                   CharLMEmbeddings('german-forward'),
                   CharLMEmbeddings('german-backward')]

# 4. init document embedding by passing list of word embeddings
document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_states=32)

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

# 6. initialize the text classifier trainer
trainer = TextClassifierTrainer(classifier, corpus, label_dict)

# 7. start the trainig
trainer.train('resources/germeval_2018/results',
              learning_rate=0.01,
              mini_batch_size=8,
              max_epochs=30,
              embeddings_in_memory=False)
