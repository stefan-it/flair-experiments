import itertools
import re
import sys

from flair.data import Sentence
from flair.models import SequenceTagger

test_file_name = sys.argv[1]

tagger: SequenceTagger = SequenceTagger.load_from_file('resources/taggers/pos-twitter-german/best-model.pt')

with open(test_file_name) as f:
    lines = [line.rstrip() for line in f.readlines()]

pos_word_pair = re.compile(r'.*type="(.*?)">(.*?)</w>')

all_sents = []
all_tags  = []

current_sentence = []
current_tags = []

for line in lines:
    if line.startswith("</tweet>"):
        all_sents.append(current_sentence)
        all_tags.append(current_tags)

        current_sentence = []
        current_tags = []

    if line.startswith("<w "):
        m = pos_word_pair.match(line)
        if m:
            current_sentence.append(m[2])
            current_tags.append(m[1])

total_counter = 0
correct_counter = 0

assert len(all_sents) == len(all_tags)

for sentence, tags in zip(all_sents, all_tags):
    sentence_str = " ".join(sentence)

    sentence_type = Sentence(sentence_str)
    tagger.predict(sentence_type)

    tokens = sentence_type.tokens
    predicted_tags = [str(token.tags['pos']).split()[0] for token in tokens]

    assert len(predicted_tags) == len(tags)

    for pred, gold in zip(predicted_tags, tags):
        total_counter += 1

        if pred == gold:
            correct_counter += 1

assert total_counter == len(list(itertools.chain(*all_tags)))

print("Accuracy:", float(correct_counter / total_counter))
