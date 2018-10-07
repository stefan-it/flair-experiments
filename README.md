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
`flair` achieves **91.89** (+ 2.47).

# Contact (Bugs, Feedback, Contribution and more)

For questions about `flair-experiments`, contact the current maintainer:
Stefan Schweter <stefan@schweter.it> or open an issue/pull request.
