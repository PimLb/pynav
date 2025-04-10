#!/bin/bash

reps=19 # number of repetitions (default 10)
out_dir="/storage/rando/pynav_results_new" # output directory (default "./results")
agent_type='two'
data_path="/storage/rando/data/nose_data_27_123.npy"
setting=("dense" "middle" "sparse"  "sparser")


for env_st in ${setting[@]};
do
for i in $(seq 0 $reps);
do
  nohup python3 train_agent.py $data_path $env_st --agent_type $agent_type --tau $1 --base_seed $i --out_path $out_dir &
done
done