#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00 --gres=gpu:k20:1
#SBATCH --job-name=sl_"${sel_lam}"__cl_"${cont_lam}"

cd /home/rcf-proj/ef/spangher/newspaper-pages/newspaper-pages/gnotebooks/text_nn
source /usr/usc/cuda/default/setup.sh

model_form=cnn
num_layers=1
word_cutoff=400
epochs=50
sel_lam="${sel_lam}"
cont_lam="${cont_lam}"

srun -n1 python3.7 -u scripts/main.py \
    --batch_size 64 \
    --cuda \
    --num_gpus 1 \
    --dataset nytimes_data \
    --embedding glove \
    --dropout 0.05 \
    --weight_decay 5e-06 \
    --num_layers ${num_layers} \
    --model_form ${model_form} \
    --hidden_dim 100 \
    --epochs ${epochs} \
    --init_lr 0.0001 \
    --num_workers 0 \
    --objective cross_entropy \
    --patience 5 \
    --save_dir snapshot \
    --train \
    --test \
    --word_cutoff ${word_cutoff} \
    --strip_punc \
    --results_path results__cont-lambda_${cont_lam}__sel-lam_${sel_lam}__word-cutoff_${word_cutoff}__epochs_${epochs}__punct-stripped  \
    --gumbel_decay 1e-5 \
    --get_rationales \
    --selection_lambda ${sel_lam} \
    --continuity_lambda ${cont_lam}
