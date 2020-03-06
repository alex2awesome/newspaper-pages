#!/bin/bash
task="qatmr"
ratio=0.8
batchsizes=(1)
mlp_hid_size=32
seed=7
model="bert-large-uncased"
pw=(3.0)
prefix="event"
ga=1
for s in "${batchsizes[@]}"
do
    learningrates=(1e-5)

    for l in "${learningrates[@]}"
    do
        epochs=( 1 )

        for e in "${epochs[@]}"
        do
	    for w in "${pw[@]}"
	    do
		python run_event_model.py \
            --task_name "${task}" \
            --do_train \
            --do_eval \
            --do_lower_case \
            --mlp_hid_size ${mlp_hid_size} \
            --model ${model} \
            --data_dir ../data/ \
            --train_ratio ${ratio} \
            --max_seq_length 130 \
            --pw ${w} \
            --train_batch_size ${s} \
            --learning_rate ${l} \
            --num_train_epochs ${e} \
            --gradient_accumulation_steps=${ga} \
            --output_dir output/${prefix}_${model}_batch_${s}_lr_${l}_epochs${e}_seed_${seed}_${ratio}_${w}_${mlp_hid_size} \
            --seed ${seed} 
	    done
	done
    done
done
