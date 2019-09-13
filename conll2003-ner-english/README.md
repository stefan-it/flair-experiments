# CoNLL-2003 NER for English

In this section, several experiments on the CoNLL-2003 NER dataset for English
are reported.

All supported Tranformer-based architectures in [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers)
are used, as well as two ELMo models.

## Changelog

**September 2019**: Experiments with stacking cased and uncased BERT models added. First results for SpanBERT are done.

## Training

The following parameters are used for all experiments:

| Parameter              | Value
| ---------------------- | -----
| `flair`                | 0.4.3
| `hidden_size`          | `256`
| `learning_rate`        | `0.1`
| `mini_batch_size`      | `16`
| `max_epochs`           | `500`

For Transformer-based architectures, all layers (incl. word embedding layer)
with scalar mix are used.

## Experiments

The following table shows all experiments on the CoNLL-2003 NER dataset.

| Model                                      | Pooling      | Dev       | Test
| ------------------------------------------ | ------------ | --------- | -----------
| BERT (base, cased)                         | `first`      | 94.74     | 91.38
| BERT (base, uncased)                       | `first`      | 94.61     | 91.03
| BERT (large, cased)                        | `first`      | 95.23     | 91.69
| BERT (large, uncased)                      | `first`      | 94.78     | 91.49
| BERT (large, cased, whole-word-masking)    | `first`      | 94.88     | 91.16
| BERT (large, uncased, whole-word-masking)  | `first`      | 94.94     | 91.20
| RoBERTa (base)                             | `first`      | 95.35     | 91.51
| RoBERTa (large)                            | `first`      | 95.83     | 92.11
| RoBERTa (large)                            | `mean`       | **96.31** | **92.31**
| XLNet (base)                               | `first_last` | 94.56     | 90.73
| XLNet (large)                              | `first_last` | 95.47     | 91.49
| XLNet (large)                              | `first`      | 95.14     | 91.71
| XLM (en)                                   | `first_last` | 94.31     | 90.68
| XLM (en)                                   | `first`      | 94.00     | 90.73
| GPT-2                                      | `first_last` | 91.35     | 87.47
| GPT-2 (large)                              | `first_last` | 94.09     | 90.63
| ELMo (original)                            | -            | 95.39     | 92.11
| ELMo (large, 5.5B tokens)                  | -            | 95.68     | 92.15
| DistilBERT (base, uncased)                 | `first`      | 94.20     | 90.68
| SpanBERT (base, cased) **preliminary**     | `first`      | 89.26     | 83.55

**Notice**: Only **one** run is reported here.

## Stacking Experiments

Stacking in this context means a concatenation of two or more embeddings from different
language models for each token. I made two experiments with "stacking" cased and uncased
BERT models:

| Models                                     | Pooling | Dev       | Test
| ------------------------------------------ | ------- | --------- | -----
| BERT (base, cased), BERT (base, uncased)   | `first` | 96.15     | 92.30
| BERT (large, cased), BERT (large, uncased) | `first` | **96.26** | 92.30

## Experiment runner

For several experiments, json-based configuration files exist. An experiment can be launched
with the `run_experiment.py` script. It provides the following commandline interface:

```bash
$ python run_experiment.py --help
Usage: run_experiment.py [OPTIONS]

Options:
  --config TEXT     Define path to configuration file
  --number INTEGER  Define experiment number
  --help            Show this message and exit.
```

E.g. to run the RoBERTa (large, mean pooling) experiment, just execute:

```bash
$ python run_experiment.py --number 1 --config configs/roberta_large_mean.json
```

Training will be automatically started with the provided parameters
(specified in the json-based configuration file).