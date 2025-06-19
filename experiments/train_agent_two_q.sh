#!/bin/bash

reps=19 # number of repetitions (default 10)

out_dir="/storage/rando/pynav_results_new" # output directory (default "./results")
#out_dir="/storage/rando/pynav_results_full_field" # output directory (default "./results")

agent_type='two'
data_path="/storage/rando/data/nose_data_27_123.npy"
#data_path="/storage/rando/data/nose_data.npy"
setting=("dense" "middle" "sparse"  "sparser")


source_location="13,0"
source_radius=2

for env_st in ${setting[@]};
do
for i in $(seq 0 $reps);
do
  nohup python3 train_agent.py $data_path $env_st $source_location $source_radius --agent_type $agent_type --tau $1 --base_seed $i --out_path $out_dir &
done
done