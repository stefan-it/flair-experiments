# Fine-grained POS Tagging of German Tweets

Abstract of the [Fine-grained POS Tagging of German Tweets](https://pdfs.semanticscholar.org/82c9/90aa15e2e35de8294b4a721785da1ede20d0.pdf)
paper from Ines Rehbein:

> This paper presents the first work on POS tagging German Twitter data, showing
> that despite the noisy and often cryptic nature of the data a fine-grained
> analysis of POS tags on Twitter microtext is feasible. Our CRF-based tagger
> achieves an accuracy of around 89% when trained on LDA word clusters, features
> from an automatically created dictionary and additional out-of-domain training
> data.

Now a model is trained, which achieves a new state-of-the-art result for the
Twitter test set 😎

## Data

The data set can be obtained from Ines Rehbein. Just use the email adresse
in the paper. She is very friendly and helpful!

The data comes in an archive `twitter_gold.tgz`. Just copy it into the
`pos-twitter-german` located in this repository. Then untar it with:

```bash
tar -xzf twitter_gold.tgz
```

Change to the extracted folder `twitter_gold`:

```bash
cd twitter_gold
```

The following xml files are relevant for training, finetuning and testing the
model:

* `twitter.gold.train.xml`
* `twitter.gold.dev.xml`
* `twitter.gold.test.xml`

Thus, we need to convert these xml files into a `flair` compatible format.

## `flair`-compatible format

The `preprocess.py` script converts each xml file into a `flair` compatible
format (just a simple CoNLL tab-separated format):

```bash
python preprocess.py twitter_gold/twitter.gold.train.xml > train.txt
python preprocess.py twitter_gold/twitter.gold.dev.xml > dev.txt
python preprocess.py twitter_gold/twitter.gold.test.xml > test.txt
```

## Training

### Experiment 1

For the first experiment we use the following parameters:

| Parameter              | Value
| ---------------------- | -----
| `flair`                | beffa4a32947d0a7a0afbb431bb65e201e4ac757 + own accuracy calculation fix
| `WordEmbeddings`       | `de-fasttext`
| `CharLMEmbeddings`     | `german-forward`
| `CharLMEmbeddings`     | `german-backward`
| `hidden_size`          | `256`
| `learning_rate`        | `0.1`
| `mini_batch_size`      | `32`
| `max_epochs`           | `150`

To reproduce the first experiment, just use the following training script:

```bash
python train_1.py
```

## Evaluation

There's no official evaluation script available. Thus, we measure the
accuracy by comparing each predicted tag in a sentence with the gold tag from
the test set.

This can be automatically done with the `predict.py` script:

```bash
python predict.py twitter_gold/twitter.gold.test.xml
```

# Results

## Task 1

### Experiment 1

For the first experiment on Task 1 the following results could be achieved:

```text
Accuracy: 0.9249629529839688
```

#### Plot

Accuracy over epochs:

![accuracy over epochs](training_1.png)

## Overview

| System                          | Final F-Score
| ------------------------------- | ---------------------------
| Best reported accuracy in paper | 89.42
| Experiment 1                    | **92.49**
