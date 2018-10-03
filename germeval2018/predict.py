import sys

from flair.data import Sentence
from flair.models import TextClassifier

tagger = TextClassifier.load_from_file('resources/germeval_2018/results/final-model.pt')

test_filename = sys.argv[1]

with open(test_filename, 'rt') as f:
  lines = [line.rstrip() for line in f.readlines()]

for line in lines:
  sentence, label, x = line.split("\t")

  new_line = [token.replace('#', '') for token in sentence.split() if not token[0] in ['@', '&', '|']]

  sentence = " ".join(new_line)

  s = Sentence(sentence, use_tokenizer=True)

  tagger.predict(s)

  label = str(s.labels[0]).split()[0]

  print(f"{sentence}\t{label}\tNOT USED")
