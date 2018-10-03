# GermEval 2018 - Task 1

## Data

The GermEval 2018 data can be cloned from the [offical](https://github.com/uds-lsv/GermEval-2018-Data)
repository:

```bash
git clone https://github.com/uds-lsv/GermEval-2018-Data
```

## Development data

The original GermEval 2018 task did not include any development data.
In our experiments we use the last 1000 lines of the training data as
development data (but we do not remove delopment data from the training data):

```bash
tail -n 1000 GermEval-2018-Data/germeval2018.training.txt > GermEval-2018-Data/germeval2018.dev.txt
```

## `flair`-compatible format

In the next step we convert the training and development data into a
`flair`-compatible format. The original data format for GermEval 2018 is
tab separated with the following columns with information:

* sentence
* label for task 1
* label for task 2

The `flair` training format is the following:

```txt
__label__<LABEL> Text input
```

Thus, the following line from the original GermEval data:

```txt
Liebe Corinna, wir würden dich gerne als Moderatorin für uns gewinnen! Wärst du begeisterbar? OTHER OTHER
```

is converted into:

```txt
__label__OTHER @corinnamilborn Liebe Corinna, wir würden dich gerne als Moderatorin für uns gewinnen! Wärst du begeisterbar?
```

This can be done in the following steps:

```bash
cat GermEval-2018-Data/germeval2018.training.txt | awk -F '[\t]' '{ print "__label__" $2,$1 }' > training.txt
cat GermEval-2018-Data/germeval2018.dev.txt | awk -F '[\t]' '{ print "__label__" $2,$1 }' > dev.txt
cat GermEval-2018-Data/germeval2018.test.txt | awk -F '[\t]' '{ print "__label__" $2,$1 }' > test.txt
```

## Preprocessing

The following preprocessing steps are done with the training and development
data:

* Remove all "#" in hashtags
* Remove all tokens that starts with "@", "|" or '&'

For that purpose, the `preprocess.py` script can be used. It takes an input
file as argument and outputs the preprocessed file on stdout:

```bash
python preprocess.py training.txt > training.preprocessed.txt
python preprocess.py dev.txt > dev.preprocessed.txt
python preprocess.py test.txt > test.preprocessed.txt
```

## Training

### Experiment 1

For the first experiment we use the following parameters:

| Parameter              | Value
| ---------------------- | -----
| `flair`                | ce1778c6c4ff7a480ff5f484f87a81d3769a5871
| `WordEmbeddings`       | `de-fasttext`
| `CharLMEmbeddings`     | `german-forward`
| `CharLMEmbeddings`     | `german-backward`
| `hidden_states`        | `32`
| `learning_rate`        | `0.01`
| `mini_batch_size`      | `8`
| `max_epochs`           | `30`
| `embeddings_in_memory` | `False`

To reproduce the first experiment, just use the following training script:

```bash
python train_1.py
```

## Evaluation

### Prediction

After training a model, the model is located under the
`resources/germeval_2018/results/final-model.pt`. The `predict.py` script reads
in the model and the test data and outputs the classification result for each
model on stdout:

```bash
python predict.py GermEval-2018-Data/germeval2018.test.txt > system.out
```

**Notice:** We use the original test data here (not the `flair`-compatible
format).

### Official evaluation

We then use the official evaluation script to verify our results:

```bash
cd GermEval-2018-Data
perl evaluationScriptGermeval2018.pl --pred ../system.out --gold germeval2018.test.txt --task 1
```

# Results

## Task 1

### Experiment 1

For the first experiment on Task 1 the following results could be achieved:

```text
Evaluate on TASK1!

****************************TASK 1: COARSE ************************
ACCURACY: 77.49 (correct=2737; total instances=3532)
CATEGORY "OFFENSE": precision=69.55 recall=60.23 fscore=64.56
CATEGORY "OTHER": precision=80.81 recall=86.39 fscore=83.51
AVERAGE: precision=75.18 recall=73.31 fscore=74.24
*******************************************************************
```

The model can be downloaded
[here](https://schweter.eu/cloud/flair-experiments/germeval2018-experiment-1-final-model.tar.xz) soon.

## Overview

| System                       | Final F-Score
| ---------------------------- | ---------------------------
| GermEval 2018 winning system | **76.77**
| Experiment 1                 | 74.24
