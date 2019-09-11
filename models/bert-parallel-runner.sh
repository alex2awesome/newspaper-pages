#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00 --gres=gpu:k40:2
#SBATCH --job-name=bert
#SBATCH --exclusive

# #Define module command, etc
# . ~/.profile
# #Load the pytorch module
# module load pytorch/0.4.1

#Number of processes per node to launch (20 for CPU, 2 for GPU)
NPROCS_PER_NODE=2

cd  /home/rcf-proj/ef/spangher/newspaper-pages/models/pytorch-transformers/examples
source /usr/usc/cuda/default/setup.sh

COMMAND="python3.7 run_glue.py \
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
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --cache_dir /home/rcf-proj/ef/spangher/newspaper-pages/models/.cache/torch"


#We want names of master and slave nodes
MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $SLAVES"

#Get a random unused port on this host(MASTER) between 2000 and 9999
#First line gets list of unused ports
#2nd line restricts between 2000 and 9999
#3rd line gets single random port from the list
MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
	grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
	sort | uniq | shuf`

#Launch the pytorch processes, first on master (first in $HOSTLIST) then
#on the slaves
RANK=0
for node in $HOSTLIST; do
	ssh -q $node \
		pytorch -m torch.distributed.launch \
		--nproces_per_node=$NPROCS_PER_NODE \
		--nnodes=$SLURM_JOB_NUM_NODES \
		--node_rank=$RANK \
		--master_addr="$MASTER" --master_port="$MPORT" \
		$COMMAND &
	RANK=$((RANK+1))
done
wait