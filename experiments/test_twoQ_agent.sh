#!/bin/bash

agent_dir="/storage/rando/pynav_results_new" # output directory (default "./results")
data_path="/storage/rando/data/nose_data_27_123.npy"
setting=("dense" "middle" "sparse" "sparser")
seeds=(1259 1574 1889 2204 2519 2834 314 3149 3464 3779 4094 4409 4724 5039 5354 5669 5984 629 6299 944 )

for env_st in ${setting[@]};
do
for seed in ${seeds[@]};
do
  nohup python3 test_agent.py $data_path $env_st $1 --agent_type "two" --tau $2  --agent_seed $seed --agent_path $agent_dir &
done
done