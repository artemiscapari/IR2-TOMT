# Setup the Conda Environment
```bash
conda env create -f environment.yml
conda activate tomt2
pip install torch
pip install transformers
conda install nltk
pip install sentence-transformers
pip install rank_bm25
```

# Prepare Data
Run the following lines to unzip the data:
```bash
unzip data/Books.zip -d data/Books
unzip data/Movies.zip -d data/Movies
```

# Reproduce Results
In order to reproduce our results run the following
```bash
sbatch run.job
```
During finetuning, three folders will appear named [evaluation](evaluation), [train_evaluation](train_evaluation), and [test_evaluation](test_evaluation), where the evaluation results will appear for each finetuned model.

# Training
Both the bi-encoder as well as the cross-encoder can be fine-tuned by running the [train.py](train.py) file.

The training function in train.py takes the following arguments:
```bash
"""
:param model_name: name of a pre-trained bi-encoder from https://huggingface.co/models?library=sentence-transformers of a pre-trained cross-encoder from https://huggingface.co/cross-encoder.
:param loss_fn:  loss function.
:param epochs: number of  epochs
:param warmup_steps: warmup step
:param learn_rate: learning rate passed
:param bm25_topk: number of retrieved bm25 results
:param train_size:number of positive samples of the training set
:param eval_size: number of positive samples of the evaluation set
:param test_size: number of positive samples of the test set
:param train_eval_size: number of positive samples of the training set to evaluate on
:param evals_per_epoch: number of times the evaluation is performed per epoch
"""
```

## Train Bi-Encoder
Any pre-trained bi-encoder from [sentence-transformers](https://huggingface.co/models?library=sentence-transformers) can be used to fine-tune our model by passing the name of the SentenceTransformer model as the argument 'model_name'.

An example of how to run the bi encoder:
```bash
conda activate tomt2
python train.py --model_name sentence-transformers/all-MiniLM-L6-v2 --epochs 25 --train_size 1000 --eval_size 1300 --test_size 1300 --train_eval_size 1000 --evals_per_epoch 5 --bm25_topk 100 --loss_fn multi-neg
```

## Train Cross-Encoder
Any pre-trained cross-encoder from [huggingface](https://huggingface.co/cross-encoder) can be used to fine-tune our model by passing the name of the SentenceTransformer model as the argument 'model_name'.

An example of how to run a Cross-Encoder:
```bash
conda activate tomt2
python train.py --model_name cross-encoder/ms-marco-TinyBERT-L-2-v2 --train_size 5000 --eval_size 50 --test_size 50 --train_eval_size 50 --evals_per_epoch 5 --bm25_topk 100
```

# Plotting
Plots of the evaluation results are created as follows:

```bash
conda activate tomt2
python plot.py
```

The resulting plots can be found in the [plots](plots) folder.