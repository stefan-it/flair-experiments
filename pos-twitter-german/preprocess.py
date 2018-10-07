import re
import sys

file_name = sys.argv[1]

with open(file_name) as f:
    lines = [line.rstrip() for line in f.readlines()]

pos_word_pair = re.compile(r'.*type="(.*?)">(.*?)</w>')

sents = []
tags  = []

current_sentence = []
current_tags = []

for line in lines:
    if line.startswith("</tweet>"):
        sents.append(current_sentence)
        tags.append(current_tags)

        current_sentence = []
        current_tags = []

    if line.startswith("<w "):
        m = pos_word_pair.match(line)
        if m:
            current_sentence.append(m[2])
            current_tags.append(m[1])

for sentence, tags in zip(sents, tags):
    for token, tag in zip(sentence, tags):
        print(f"{token}\t{tag}")

    print("")
