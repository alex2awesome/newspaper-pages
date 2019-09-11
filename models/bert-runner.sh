#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00 --gres=gpu:k40:2
#SBATCH --job-name=bert

cd  /home/rcf-proj/ef/spangher/newspaper-pages/models/pytorch-transformers/examples
source /usr/usc/cuda/default/setup.sh

python3.7 run_glue.py \
  --task_name sst-2 \
  --model_type bert \
  --model_name_or_path bert-large-uncased \
  --data_dir /home/rcf-proj/ef/spangher/newspaper-pages/data/bert-data/ \
  --output_dir bert-output \
  --max_seq_length 300 \
  --do_train \
  --do_eval \
  --save_steps 2500 \
  --do_lower_case \
  --local_rank 0 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --cache_dir /home/rcf-proj/ef/spangher/newspaper-pages/models/.cache/torch
