import logging
import sys

from flair import set_seed
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")

# Hyper-param search
seeds = [1, 2, 3, 4, 5]
batch_sizes = [4, 8, 16, 32]

for batch_size in batch_sizes:
    for seed in seeds:
        set_seed(seed)

        columns = {0: "text", 1: "pos"}
        corpus = ColumnCorpus(data_folder="./", column_format=columns, in_memory=True, train_file="train.txt",
                              dev_file="dev.txt", test_file="test.txt", column_delimiter="\t")

        # Training, dev and test data set size from paper
        number_train_tweets = 420
        number_dev_tweets = 500
        number_test_tweets = 506

        # Check, if our dataset reader works
        assert len(corpus.train) == number_train_tweets
        assert len(corpus.dev) == number_dev_tweets
        assert len(corpus.test) == number_test_tweets

        tag_dictionary = corpus.make_label_dictionary(label_type="pos", add_dev_test=True)

        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('de'),
            FlairEmbeddings('german-forward'),
            FlairEmbeddings('german-backward'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type="pos",
                                                use_crf=True)

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(f'./pos-twitter-german-bs{batch_size}-{seed}',
                      learning_rate=0.1,
                      mini_batch_size=batch_size,
                      main_evaluation_metric=("micro avg", "accuracy"),
                      use_final_model_for_eval=False,
                      max_epochs=150)
