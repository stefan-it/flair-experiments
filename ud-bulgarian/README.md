# Bulgarian POS tagging: Universal dependencies

We use Bulgarian UD for training a model with `flair`.

## Data

The train, dev and test datasets are used from:

<https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1548>

This is universal dependencies in version 1.2 which is mostly used in papers.


Thus, this data needs to be downloaded with:

```bash
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1548{/ud-treebanks-v1.2.tgz}
```

Extract the downloaded archive with:

```bash
tar -xzf ud-treebanks-v1.2.tgz
```

## Word Embeddings

Pretrained `fasttext` word embeddings needs to be downloaded with:

```
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.bg.vec
```

## Language models

Forward and backward language models needs also to be downloaded with:

```
wget https://schweter.eu/cloud/flair-lms/lm-bg-small-forward-v0.1.pt
wget https://schweter.eu/cloud/flair-lms/lm-bg-small-backward-v0.1.pt
```

## `flair`-compatible format

`flair` already comes with an import for the universal dependencies format.

## Training

### Experiment 1

| Parameter              | Value
| ---------------------- | -----
| `flair`                | 4a4a38e0c3b1ba71d26d379bf10918aa0dce28f3 + own accuracy calculation fix
| `WordEmbeddings`       | Pretrained Bulgarian `fasttext` embeddings from [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
| `CharLMEmbeddings`     | Forward lm from [here](https://github.com/stefan-it/flair-lms#bulgarian)
| `CharLMEmbeddings`     | Backward lm from [here](https://github.com/stefan-it/flair-lms#bulgarian)
| `hidden_size`          | `256`
| `learning_rate`        | `0.1`
| `mini_batch_size`      | `8`
| `max_epochs`           | `500`

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
python predict.py universal-dependencies-1.2/UD_Bulgarian/bg-ud-test.conllu
```

# Results

## Experiment 1

For the first experiment the following results could be achieved:

```bash
Accuracy: 0.990784288801322
```

## Overview

| System                                                     | Final Accuracy
| ---------------------------------------------------------- | -------------
| [Yu et. al (2017)](https://arxiv.org/abs/1706.01723)       | 98.20
| [Plank et. al (2016)](https://arxiv.org/abs/1604.05529)    | 98.23
| [Yasunaga et. al (2017)](https://arxiv.org/abs/1711.04903) | 98.53
| Experiment 1                                               | **99.08**
