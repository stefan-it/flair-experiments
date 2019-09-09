import sys
from pathlib import Path


import flair.datasets
from flair.data import Sentence
from flair.models import SequenceTagger

model = sys.argv[1]

ud_path = Path("universal-dependencies-1.2/UD_English")

corpus = flair.datasets.UniversalDependenciesCorpus(data_folder=ud_path)

print(corpus)

tag_type = "upos"
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

tagger: SequenceTagger = SequenceTagger.load(model)

number_tags = 0
number_correct_tags = 0

for sentence in corpus.test:
    tokens = sentence.tokens
    gold_tags = [token.tags['upos'].value for token in sentence.tokens]

    tagged_sentence = Sentence()
    tagged_sentence.tokens = tokens

    tagger.predict(tagged_sentence)

    predicted_tags = [token.tags['upos'].value for token in tagged_sentence.tokens]
    
    assert len(tokens) == len(gold_tags)
    assert len(gold_tags) == len(predicted_tags)
    
    number_tags += len(predicted_tags)
    
    for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
        if gold_tag == predicted_tag:
            number_correct_tags += 1

print(f'Accuracy: {number_correct_tags / number_tags}')
