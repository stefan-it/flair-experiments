# English POS tagging: Universal dependencies

We use English UD in version 1.2 for training a model with `flair`.

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

The following table shows all experiments on the UD v1.2 dataset for English.

| Model                                      | Pooling      | Dev       | Test
| ------------------------------------------ | ------------ | --------- | ---------
| RoBERTa (large)                            | `mean`       | **97.80** | **97.75**
| SpanBERT (large)                           | `first`      | 96.48     | 96.61
| BERT (large, cased)                        | `first`      | 97.35     | 97.20
| DistilBERT (uncased)                       | `first`      | 96.64     | 96.70

**Notice**: Only **one** run is reported here.

## SOTA

| System                                                            | Final Accuracy
| ----------------------------------------------------------------- | -------------
| [Plank et. al (2016)](https://arxiv.org/abs/1604.05529)           | 95.52
| [Yasunaga et. al (2017)](https://arxiv.org/abs/1711.04903)        | 95.82
| [Heinzerling and Strube (2019)](https://arxiv.org/abs/1906.01569) | 95.60‡

‡ refers to their MultiBPemb model (fine-tuned).
