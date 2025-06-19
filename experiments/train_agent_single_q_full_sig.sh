#!/bin/bash

reps=19 # number of repetitions (default 10)

#out_dir="/storage/rando/pynav_results_new" # output directory (default "./results")
out_dir="/storage/rando/pynav_results/full_field" # output directory (default "./results")

#data_path="/storage/rando/data/nose_data_27_123.npy"
data_path="/storage/rando/data/nose_data.npy"



setting=("dense" "middle" "sparse"  "sparser")


for env_st in ${setting[@]};
do
for i in $(seq 0 $reps);
do
  nohup python3 train_agent.py $data_path $env_st "142,0" 15 --agent_velocity 10 \
  --gamma 0.9998 --num_episodes 6000001 --horizon 5000 \
  --eps_init 1.0 --eps_end 0.0001 --eps_decay 5500000 \
  --alpha_init 0.1 --alpha_end 0.0001 --alpha_decay 5500000 \
  --checkpoint_iters 50000 --agent_type "single"  --base_seed $i --out_path $out_dir &
done
done