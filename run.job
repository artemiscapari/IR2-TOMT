#!/bin/sh

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=TrainVAE
#SBATCH --cpus-per-task=3
#SBATCH --time=45:00:00
#SBATCH --mem=64GB
#SBATCH --output=slurm_output_%A.out
#SBATCH --partition=gpu_shared_course
source ${HOME}/.bashrc

# module purge
# module load 2019
# module load Anaconda3/2018.12
# module load Python/3.7.5-foss-2019b
# module load CUDA/10.1.243
# module load cuDNN/7.6.5.32-CUDA-10.1.243
# module load NCCL/2.5.6-CUDA-10.1.243
# # GPU
# pip install torch
export CUDA_HOME="/usr/local/cuda-10.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

module load 2021
module load Java/11.0.2
conda activate tomt2

#############################################################################################
# BI-ENCODER all-MiniLM-L6-v2
#############################################################################################

python train.py --model_name sentence-transformers/all-MiniLM-L6-v2 --train_size 1000 --eval_size 1300 --test_size 1300 --train_eval_size 1000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-MiniLM-L6-v2 --train_size 5000 --eval_size 1300 --test_size 1300 --train_eval_size 5000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-MiniLM-L6-v2 --train_size 10000 --eval_size 1300 --test_size 1300 --train_eval_size 10000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-MiniLM-L6-v2 --train_size 5000 --eval_size 1300 --test_size 1300 --train_eval_size 5000 --bm25_topk 1000 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-MiniLM-L6-v2 --train_size 1000 --eval_size 1300 --test_size 1300 --train_eval_size 1000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn cos-sim


#############################################################################################
# BI-ENCODER all-mpnet-base-v2
#############################################################################################

python train.py --model_name sentence-transformers/all-mpnet-base-v2 --train_size 1000 --eval_size 1300 --test_size 1300 --train_eval_size 1000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-mpnet-base-v2 --train_size 5000 --eval_size 1300 --test_size 1300 --train_eval_size 5000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-mpnet-base-v2 --train_size 10000 --eval_size 1300 --test_size 1300 --train_eval_size 10000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-mpnet-base-v2 --train_size 5000 --eval_size 1300 --test_size 1300 --train_eval_size 5000 --bm25_topk 1000 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn multi-neg

python train.py --model_name sentence-transformers/all-mpnet-base-v2 --train_size 1000 --eval_size 1300 --test_size 1300 --train_eval_size 1000 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5 --loss_fn cos-sim


#############################################################################################
# CROSS-ENCODER ms-marco-TinyBERT-L-2-v2
#############################################################################################

python train.py --model_name cross-encoder/ms-marco-TinyBERT-L-2-v2 --train_size 1000 --eval_size 100 --test_size 100 --train_eval_size 100 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5

python train.py --model_name cross-encoder/ms-marco-TinyBERT-L-2-v2 --train_size 5000 --eval_size 100 --test_size 100 --train_eval_size 100 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5

python train.py --model_name cross-encoder/ms-marco-TinyBERT-L-2-v2 --train_size 10000 --eval_size 100 --test_size 100 --train_eval_size 100 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5

#############################################################################################
# CROSS-ENCODER ms-marco-MiniLM-L-6-v2
#############################################################################################

python train.py --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 --train_size 1000 --eval_size 100 --test_size 100 --train_eval_size 100 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5

python train.py --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 --train_size 5000 --eval_size 100 --test_size 100 --train_eval_size 100 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5

python train.py --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 --train_size 10000 --eval_size 100 --test_size 100 --train_eval_size 100 --bm25_topk 100 --eval_batch_size 200 --evals_per_epoch 5

#############################################################################################
# PLOT RESULTS
#############################################################################################

python plot.py
