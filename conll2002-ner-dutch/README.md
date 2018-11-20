# Language-Independent Named Entity Recognition (I): CoNLL 2002 for Dutch

We use the official CoNLL 2002 data for NER on Dutch.

## Data

The train, dev and test datasets are used from:

<https://www.clips.uantwerpen.be/conll2002/ner/>

Thus, this data needs to be downloaded with:

```bash
wget https://www.clips.uantwerpen.be/conll2002/ner/data/ned.train
wget https://www.clips.uantwerpen.be/conll2002/ner/data/ned.testa
wget https://www.clips.uantwerpen.be/conll2002/ner/data/ned.testb
```

As development data, `ned.testa` is used. `ned.testb` is then used as test data
for final evaluation.

Before training a model, all datasets needs to be converted from isolatin to
UTF-8:

```bash
recode l1..u8 ned.train
recode l1..u8 ned.testa
recode l1..u8 ned.testb
```

## Word Embeddings

Pretrained `fasttext` word embeddings needs to be downloaded with:

```
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.nl.vec
```

## Language models

Forward and backward language models needs also to be downloaded with:

```
wget https://schweter.eu/cloud/flair-lms/lm-nl-large-forward-v0.1.pt
wget https://schweter.eu/cloud/flair-lms/lm-nl-large-backward-v0.1.pt
```

## `flair`-compatible format

`flair` already comes with an importer for the CoNLL format.

## Training

### Experiment 1

| Parameter              | Value
| ---------------------- | -----
| `flair`                | 86485ec920e30b787c43b6b3ff094fbc4fa0c253
| `WordEmbeddings`       | Pretrained Dutch `fasttext` embeddings from [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
| `CharLMEmbeddings`     | Forward lm from [here](https://github.com/stefan-it/flair-lms#dutch)
| `CharLMEmbeddings`     | Backward lm from [here](https://github.com/stefan-it/flair-lms#dutch)
| `hidden_size`          | `256`
| `learning_rate`        | `0.1`
| `mini_batch_size`      | `8`
| `max_epochs`           | `500`

To reproduce the first experiment, just use the following training script:

```bash
python train_1.py
```

## Evaluation

We use the official evaluation script from CoNNL website. It can be downloaded
with:

```bash
wget https://www.clips.uantwerpen.be/conll2002/ner/bin/conlleval.txt
```

The `predict_ner.py` script produces the correct input format for the evaluation
script:

```bash
python predict_ner.py ned.testb > output.system
```

In the final step, the evaluation script must be called with:

```bash
perl conlleval.txt < output.system
```

# Results

## Experiment 1

The CoNNL evaluation script outputs:

```bash
processed 68875 tokens with 3941 phrases; found: 3913 phrases; correct: 3517.
accuracy:  98.96%; precision:  89.88%; recall:  89.24%; FB1:  89.56
              LOC: precision:  93.03%; recall:  94.83%; FB1:  93.92  789
             MISC: precision:  90.36%; recall:  81.30%; FB1:  85.59  1068
              ORG: precision:  83.85%; recall:  86.51%; FB1:  85.16  910
              PER: precision:  92.06%; recall:  96.08%; FB1:  94.03  1146
```

Thus, a f-score of 89.56 was achieved.

## Overview

| System                                                                          | Final Accuracy
| ------------------------------------------------------------------------------- | -------------
| CoNLL 2002 [best system](https://www.clips.uantwerpen.be/conll2002/ner/#CMP02)  | 77.05
| Experiment 1                                                                    | **89.56**


# Downloads

The trained model for experiment 1 can be downloaded with:

```bash
wget https://schweter.eu/cloud/flair-models/nl-ner-conll02-v0.1.pt
```
