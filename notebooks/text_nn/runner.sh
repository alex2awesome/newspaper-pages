#!/usr/bin/env bash
#SBATCH --ntasks=4
#SBATCH --time=3:00:00 --gres=gpu:k20:1
cd /home/rcf-proj/ef/spangher/newspaper-pages/text_nn
source /usr/usc/cuda/default/setup.sh

model_form=cnn
num_layers=1
word_cutoff=80
epochs=50

srun -n1 python3.7 -u scripts/main.py --batch_size 64 --cuda --num_gpus 1 --dataset nytimes_data --embedding glove --dropout 0.05 --weight_decay 5e-06 --num_layers ${num_layers} --model_form ${model_form} --hidden_dim 100 --epochs ${epochs} --init_lr 0.0001 --num_workers 0 --objective cross_entropy --patience 5 --save_dir snapshot --train --test --results_path results__model-form_${model_form}__num-layers_${num_layers}__word-cutoff_${word_cutoff}__epochs_${epochs}  --gumbel_decay 1e-5 --get_rationales --selection_lambda .001 --continuity_lambda 0
