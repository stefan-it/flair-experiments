import sys

from pathlib import Path

import flair.datasets
from flair.embeddings import FlairEmbeddings, BertEmbeddings, RoBERTaEmbeddings, OpenAIGPTEmbeddings, OpenAIGPT2Embeddings, XLMEmbeddings, TransformerXLEmbeddings, XLNetEmbeddings, StackedEmbeddings, ELMoEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

model_name: str = sys.argv[1]
layers: str = sys.argv[2]

description = f"{model_name}_{layers}_bs16"
hidden_size = 256
batch_size = 16

ud_path = Path("universal-dependencies-1.2/UD_English")

corpus = flair.datasets.UniversalDependenciesCorpus(data_folder=ud_path)

print(corpus)

tag_type = "upos"
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

print(f"Model name: {model_name}")
print(f"Layers: {layers}")

embeddings: StackedEmbeddings = []

if model_name == "roberta_large_mean":
    emb = RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-large", layers=layers, pooling_operation="mean", use_scalar_mix=True)
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=[emb])
elif model_name == "spanbert_large":
    emb = BertEmbeddings(bert_model_or_path="/mnt/spanbert-large-cased", layers=layers, use_scalar_mix=True)
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=[emb])
elif model_name == "bert_large_cased":
    emb = BertEmbeddings(bert_model_or_path="bert-large-cased", layers=layers, use_scalar_mix=True)
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=[emb])
elif model_name == "distilbert_base_uncased":
    emb = BertEmbeddings(bert_model_or_path="distilbert-base-uncased", layers=layers, use_scalar_mix=True)
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=[emb])

tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train(f'resources/taggers/experiment_{description}',
              learning_rate=0.1,
              mini_batch_size=batch_size,
              max_epochs=500,
              embeddings_storage_mode='gpu',
              train_with_dev=False,
              )
