import sys 
import os
import argparse as ap
import numpy as np 

from pynav import QAgent, TwoQAgent, Environment
from pynav.simulators import Simulator

settings = {
    'dense' : 1e-6,
    'middle' : 3e-6,
    'sparse' : 5e-6,
    'sparser' : 7e-6
}

actions = {
    0 : np.asarray([ 1, 0], dtype=np.int32), 
    1 : np.asarray([-1, 0], dtype=np.int32), 
    2 : np.asarray([0,  1], dtype=np.int32), 
    3 : np.asarray([0, -1], dtype=np.int32) 
}

seeds = {
    'agent_seed' : 314,
    'env_seed' : 154,
    'samp_seed' : 980,
    'sim_seed' : 723
}

horizon = 500 #1000

gamma = 0.998 # Because the horizon is 500

source_position = np.array([13, 0])
source_radius = 2

# IT WORKS!
# USED FOR SINGLE Q
# num_episodes = 3000001
# eps_init, eps_end, eps_decay = 1.0, 0.0001, 2000000 
# alpha_init, alpha_end, alpha_decay = 0.01, 0.0001, 2000000


# FINAL CONFIGURATION FOR BOTH AGENTS

num_episodes = 3000001
eps_init, eps_end, eps_decay = 1.0, 0.0001, 2500000 
alpha_init, alpha_end, alpha_decay = 0.1, 0.0001, 2500000


# CURRENTLY USED FOR TWO Q
# num_episodes = 3000001
# eps_init, eps_end, eps_decay = 1.0, 0.0001, 2000000 
# alpha_init, alpha_end, alpha_decay = 0.01, 0.00001, 2500000

# num_episodes = 3000001
# eps_init, eps_end, eps_decay = 1.0, 0.0001, 2000000 
# alpha_init, alpha_end, alpha_decay = 0.1, 0.00001, 2500000




eps_greedy = lambda t : eps_end + ( eps_init - eps_end) * max((eps_decay - t - 1)/eps_decay, 0)
learning_rate = lambda t : alpha_end + ( alpha_init - alpha_end) * max((alpha_decay - t - 1)/alpha_decay, 0)


checkpoint_iters = 10000


def train_agent(args):
    n_thr = settings[args.setting]
    data = np.load(args.data_path)
    
    for key in seeds.keys():
        seeds[key] =(args.base_seed + 1) * seeds[key] + args.base_seed
        
    
    out_dir = f"{args.out_path}/{args.agent_type}_Q/{args.setting}/agent_{seeds['agent_seed']}"
    
    if args.agent_type == 'two':
        out_dir += f"/tau_{args.tau}"
    
    rnd_state = np.random.RandomState(seeds['samp_seed'])
    os.makedirs(f"{out_dir}/Q_stored", exist_ok=True)
    
    start_points = np.any(data > 7e-6, axis=0).astype(np.float64)
    start_points[:, :3] = 0.0 # Set to 0.0 the probability to sample initial conditions in the source region
    start_points /= np.sum(start_points)
    start_points = start_points.ravel()
    
    data[data < n_thr] = 0.0
    
    
    def sample_init_position():
        pos_idx = rnd_state.choice(range(start_points.shape[0]), 1, p=start_points)
        pos_y, pos_x = np.unravel_index(pos_idx, shape=(data.shape[1], data.shape[2]))
        pos = np.concatenate([pos_y, pos_x])
        pos = pos.astype(int)
        return pos    
    env = Environment(data, 
                      source_position=source_position, 
                      source_radius=source_radius,
                      gamma = gamma,
                      seed = seeds['env_seed'])

    if args.agent_type == 'single':
        agent = QAgent(
            horizon=horizon,
            gamma=gamma,
            eps_greedy= eps_greedy,
            learning_rate= learning_rate,
            seed=seeds['agent_seed'],
            s_thr=n_thr,
            out_dir=f"{out_dir}/Q_stored"
        )
    else:
        agent = TwoQAgent(
            horizon=horizon,
            gamma=gamma,
            eps_greedy= eps_greedy,
            learning_rate= learning_rate,
            seed=seeds['agent_seed'],
            tau=args.tau,
            s_thr=n_thr,
            out_dir=f"{out_dir}/Q_stored"
        )
    
    simulator = Simulator(env=env, actions=actions)

    cumulative_rewards, agent_speed, convergence, max_void_steps, num_steps, max_void_lengths, init_positions = simulator.train_agent(agent, 
                                                                        point_sampler=sample_init_position, 
                                                                        horizon=horizon, 
                                                                        num_episodes=num_episodes, 
                                                                        delta=500, 
                                                                        checkpoint_iters=checkpoint_iters, 
                                                                        seed=seeds['sim_seed'])
    agent.store_q("Qfun")
    with open(f"{out_dir}/training.log", 'w') as f:
        for i in range(len(cumulative_rewards)):
            f.write("{},{},{},{},{},{},{}\n".format(init_positions[i][0], init_positions[i][1], cumulative_rewards[i], agent_speed[i], convergence[i], num_steps[i], max_void_steps[i]))
    np.save(f"{out_dir}/training_max_void_lengths", np.array(max_void_lengths))




if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Training QAgent.", formatter_class=ap.ArgumentDefaultsHelpFormatter)

    # Data parameters
    parser.add_argument("data_path", type=str, help="path to ,file containing data")
    parser.add_argument("setting", type=str, choices=['dense', 'middle', 'sparse', 'sparser'], help='setting considered')


    # Experiment parameters
    parser.add_argument("--agent_type", type=str, choices=['single', 'two'], default='single', help="Type of agent. If 'single' it uses a single Q matrix if 'two' it uses two Q")
    parser.add_argument("--tau", type=int, default=1, help='Threshold for two Q agent')

    parser.add_argument("--base_seed", type=int, default=143125, help='Seed used by the agent for epsilon-greedy strategy')
    parser.add_argument("--out_path", type=str, default='./results', help='output directory')
    
    args = parser.parse_args()
    train_agent(args)