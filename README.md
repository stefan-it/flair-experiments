# `flair-experiments`

This repository is part of my NLP research with
[`flair`](https://github.com/zalandoresearch/flair), a state-of-the-art NLP
framework from [Zalando Research](https://research.zalando.com/).

This repository will include models for various NLP benchmarks, such as
GermEval 2018. It will be updated frequently. So please star or watch this
repository 😅

# GermEval 2018

## Task 1

The first task of GermEval 2018 is to decide whether a tweet includes a) some
form of offensive language or b) or not.

All details for training a model with `flair` and achieving state-of-the-art
results are located in the [GermEval 2018](germeval2018/README.md) readme.

The winning system for task 1 achieved a F-Score of 76.77. The currently best
model trained with `flair` achieves a F-Score from **74.24**.

# Fine-grained POS Tagging of German Tweets

All details for training a model with `flair` and achieving a new
state-of-the-art result for the paper
[Fine-grained POS Tagging of German Tweets](https://pdfs.semanticscholar.org/82c9/90aa15e2e35de8294b4a721785da1ede20d0.pdf)
are located in the [POS Twitter German](pos-twitter-german/README.md) readme.

The paper reported an accuracy of 89.42. The currently best model trained with
`flair` achieves **92.49** (+ 3.07).

# German Universal Dependencies 1.2

All details for training a model with `flair` on German universal dependencies
and achieving a new state-of-the-art result can be found in the
[UD German](ud-german/README.md) readme.

The current state-of-the-art result for German UD is reported by
[Yasunaga et. al (2017)](https://arxiv.org/abs/1711.04903). They use
adversarial training and their system achieves an accuracy of 94.35. With `flair`
an accuracy of **94.52** (+ 0.17) can be achieved.

# Bulgarian Universal Dependencies 1.2

All details for training a model with `flair` on Bulgarian universal
dependencies and achieving a new state-of-the-art result can be found in the
[UD Bulgarian](ud-bulgarian/README.md) readme.

The current state-of-the-art result for Bulgarian UD is reported by
[Yasunaga et. al (2017)](https://arxiv.org/abs/1711.04903). They use
adversarial training and their system achieves an accuracy of 98.53. With `flair`
an accuracy of **99.08** (+ 0.55) can be achieved.

# Contact (Bugs, Feedback, Contribution and more)

For questions about `flair-experiments`, contact the current maintainer:
Stefan Schweter <stefan@schweter.it> or open an issue/pull request.
