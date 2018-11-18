import itertools
import re
import sys

from flair.data import Sentence
from flair.models import SequenceTagger

test_file_name = sys.argv[1]

tagger: SequenceTagger = SequenceTagger.load_from_file('resources/taggers/ud-slovenian/best-model.pt')

with open(test_file_name) as f:
    lines = [line.rstrip() for line in f.readlines()]

all_sents = []
all_tags  = []

current_sentence = []
current_tags = []

for line in lines:
    if len(line) == 0:
        all_sents.append(current_sentence)
        all_tags.append(current_tags)

        current_sentence = []
        current_tags = []
    elif line.startswith("#"):
      continue
    else:
        # 2       nedavnem        nedaven ADJ     Agpmsl  Case=Loc|Degree=Pos|Gender=Masc|Number=Sing     4       amod    _       Dep=4|Rel=Atr
        line_splitted = line.split("\t")
        id_ = line_splitted[0]
        token = line_splitted[1]
        upos = line_splitted[3]

        if "." in id_ or "-" in id_:
            continue

        current_sentence.append(token)
        current_tags.append(upos)

total_counter = 0
correct_counter = 0

assert len(all_sents) == len(all_tags)

for sentence, tags in zip(all_sents, all_tags):
    sentence_str = " ".join(sentence)

    sentence_type = Sentence(sentence_str)
    tagger.predict(sentence_type)

    tokens = sentence_type.tokens
    predicted_tags = [str(token.tags['upos']).split()[0] for token in tokens]

    assert len(predicted_tags) == len(tags)

    for pred, gold in zip(predicted_tags, tags):
        total_counter += 1

        if pred == gold:
            correct_counter += 1

assert total_counter == len(list(itertools.chain(*all_tags)))

print("Accuracy:", float(correct_counter / total_counter))
print("Correct counter:", correct_counter)
print("Total counter:", total_counter)
print("All sents:", len(all_sents))
