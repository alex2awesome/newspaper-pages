#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00 --gres=gpu:p100:1
#SBATCH --job-name=rob-hf

cd  /home/rcf-proj/ef/spangher/newspaper-pages/models/pytorch-transformers/examples
source /usr/usc/cuda/default/setup.sh

DATA_DIR=/home/rcf-proj/ef/spangher/newspaper-pages/data

python3.7 run_glue.py \
  --task_name sst-2 \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --data_dir $DATA_DIR/bert-data/ \
  --output_dir roberta-hf-output \
  --max_seq_length 300 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --cache_dir /home/rcf-proj/ef/spangher/newspaper-pages/models/.cache/torch
