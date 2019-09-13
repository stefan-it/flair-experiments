import json

import flair.datasets
from dataclasses import dataclass

from flair import logger
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    BertEmbeddings,
    ELMoEmbeddings,
    OpenAIGPT2Embeddings,
    TokenEmbeddings,
    RoBERTaEmbeddings,
    StackedEmbeddings,
    XLMEmbeddings,
    XLNetEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from typing import Dict, List, Tuple, Union


@dataclass
class Experiment:
    description: str
    embeddings: List[str]
    layers: List[int]
    batch_size: int
    hidden_size: int
    max_epochs: int
    embeddings_storage_mode: str
    pooling_operation: str
    use_crf: bool
    use_scalar_mix: bool
    train_with_dev: bool


class ExperimentRunner:
    def __init__(self, number: int, configuration_file: str):
        self.experiment = Experiment(**self._get_experiment_details(configuration_file))
        self.number = number

        logger.info(self.experiment)

    def start(self) -> None:
        self.stacked_embeddings = self._get_stacked_embeddings()
        description = self.experiment.description.replace(" ", "_")
        batch_size = self.experiment.batch_size
        max_epochs = self.experiment.max_epochs
        embeddings_storage_mode = self.experiment.embeddings_storage_mode
        train_with_dev = self.experiment.train_with_dev

        tagger, corpus = self._get_sequence_tagger()

        trainer = ModelTrainer(tagger, corpus)

        trainer.train(
            f"resources/taggers/experiment_{description}_{self.number}",
            learning_rate=0.1,
            mini_batch_size=batch_size,
            max_epochs=max_epochs,
            embeddings_storage_mode=embeddings_storage_mode,
            train_with_dev=train_with_dev,
        )

    def _get_experiment_details(
        self, configuration_file: str
    ) -> Dict[str, Union[str, bool, int]]:
        with open(configuration_file, "r") as f_p:
            return json.load(f_p)

    def _get_stacked_embeddings(self) -> StackedEmbeddings:
        layers = ",".join(str(layer) for layer in self.experiment.layers)
        pooling_operation = self.experiment.pooling_operation

        token_embeddings = []

        for embedding in self.experiment.embeddings:
            if embedding.startswith("roberta"):
                token_embeddings.append(
                    RoBERTaEmbeddings(
                        pretrained_model_name_or_path=embedding,
                        pooling_operation=pooling_operation,
                        layers=layers,
                        use_scalar_mix=self.experiment.use_scalar_mix,
                    )
                )
            elif (
                embedding.startswith("bert")
                or embedding.startswith("distilbert")
                or embedding.startswith("spanbert")
            ):
                token_embeddings.append(
                    BertEmbeddings(
                        bert_model_or_path=embedding,
                        pooling_operation=pooling_operation,
                        layers=layers,
                        use_scalar_mix=self.experiment.use_scalar_mix,
                    )
                )
            elif embedding.startswith("elmo"):
                model_name = embedding.split("-")[-1]
                token_embeddings.append(ELMoEmbeddings(model=model_name))
            elif embedding.startswith("gpt2"):
                token_embeddings.append(
                    OpenAIGPT2Embeddings(
                        pretrained_model_name_or_path=embedding,
                        pooling_operation=pooling_operation,
                        layers=layers,
                        use_scalar_mix=self.experiment.use_scalar_mix,
                    )
                )
            elif embedding.startswith("xlm"):
                token_embeddings.append(
                    XLMEmbeddings(
                        pretrained_model_name_or_path=embedding,
                        pooling_operation=pooling_operation,
                        layers=layers,
                        use_scalar_mix=self.experiment.use_scalar_mix,
                    )
                )
            elif embedding.startswith("xlnet"):
                token_embeddings.append(
                    XLNetEmbeddings(
                        pretrained_model_name_or_path=embedding,
                        pooling_operation=pooling_operation,
                        layers=layers,
                        use_scalar_mix=self.experiment.use_scalar_mix,
                    )
                )

        return StackedEmbeddings(embeddings=token_embeddings)

    def _get_sequence_tagger(self) -> Tuple[SequenceTagger, ColumnCorpus]:
        corpus = flair.datasets.CONLL_03()
        tag_type = "ner"
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        tagger = SequenceTagger(
            hidden_size=self.experiment.hidden_size,
            embeddings=self.stacked_embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=self.experiment.use_crf,
        )

        return tagger, corpus
