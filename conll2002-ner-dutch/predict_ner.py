import itertools
import re
import sys

from flair.data import Sentence
from flair.models import SequenceTagger

test_file_name = sys.argv[1]

tagger: SequenceTagger = SequenceTagger.load_from_file('resources/taggers/ner-dutch/best-model.pt')

with open(test_file_name) as f:
    lines = [line.rstrip() for line in f.readlines()]

all_sents = []
all_tags  = []

current_sentence = []
current_tags = []

for line in lines:
    if len(line) == 0:
        sentence_str = " ".join(current_sentence)

        sentence_type = Sentence(sentence_str)
        tagger.predict(sentence_type)

        tokens = sentence_type.tokens
        predicted_tags = [str(token.tags['ner']).split()[0] for token in tokens]

        assert len(current_sentence) == len(current_tags)
        assert len(current_tags) == len(predicted_tags)

        for token, gold_tag, predicted_tag in zip(current_sentence, current_tags, predicted_tags):

            if predicted_tag.startswith('S-'):
                predicted_tag = predicted_tag.replace('S-', 'B-')

            if predicted_tag.startswith('E-'):
                predicted_tag = predicted_tag.replace('E-', 'I-')

            # Use dummy pos tag ;)
            print(f'{token}\tPOS\t{gold_tag}\t{predicted_tag}')

        print("")

        current_sentence = []
        current_tags = []
    elif line.startswith("#") or line.startswith('-DOCSTART-'):
        continue
    else:
        # Engeland N B-LOC
        line_splitted = line.split()
        token = line_splitted[0]
        pos = line_splitted[1]
        ner = line_splitted[2]

        current_sentence.append(token)
        current_tags.append(ner)
