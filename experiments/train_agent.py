import sys 
import os
import argparse as ap
import numpy as np 

from pynav import QAgent, TwoQAgent, Environment
from pynav.simulators import Simulator

from utils import parse_numpy_array, write_info_file

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

#horizon = 500 # for small field

# horizon = 5000 # for full field



# gamma = 0.998 # for small field

# # gamma = 0.9998 # for full field


# source_position = np.array([13, 0])
# source_radius = 2

# source_position = np.array([142, 0])
# source_radius = 15 #20


# IT WORKS!
# USED FOR SINGLE Q
# num_episodes = 3000001
# eps_init, eps_end, eps_decay = 1.0, 0.0001, 2000000 
# alpha_init, alpha_end, alpha_decay = 0.01, 0.0001, 2000000


# FINAL CONFIGURATION FOR BOTH AGENTS

# num_episodes = 3000001
# eps_init, eps_end, eps_decay = 1.0, 0.0001, 2500000 
# alpha_init, alpha_end, alpha_decay = 0.1, 0.0001, 2500000

# num_episodes = 6000001
# eps_init, eps_end, eps_decay = 1.0, 0.0001, 5500000 
# alpha_init, alpha_end, alpha_decay = 0.1, 0.0001, 5500000



# # CURRENTLY USED FOR TWO Q
# # num_episodes = 3000001
# # eps_init, eps_end, eps_decay = 1.0, 0.0001, 2000000 
# # alpha_init, alpha_end, alpha_decay = 0.01, 0.00001, 2500000

# # num_episodes = 3000001
# # eps_init, eps_end, eps_decay = 1.0, 0.0001, 2000000 
# # alpha_init, alpha_end, alpha_decay = 0.1, 0.00001, 2500000




# eps_greedy = lambda t : eps_end + ( eps_init - eps_end) * max((eps_decay - t - 1)/eps_decay, 0)
# learning_rate = lambda t : alpha_end + ( alpha_init - alpha_end) * max((alpha_decay - t - 1)/alpha_decay, 0)


#checkpoint_iters = 10000


def train_agent(args):
    
    # Get noise threshold and load data
    n_thr = settings[args.setting]
    data = np.load(args.data_path)

    # Exploration/Exploitation parameters and learning rate
    eps_init, eps_end, eps_decay = args.eps_init, args.eps_end, args.eps_decay
    alpha_init, alpha_end, alpha_decay = args.alpha_init, args.alpha_end, args.alpha_decay
    eps_greedy = lambda t : eps_end + ( eps_init - eps_end) * max((eps_decay - t - 1)/eps_decay, 0)
    learning_rate = lambda t : alpha_end + ( alpha_init - alpha_end) * max((alpha_decay - t - 1)/alpha_decay, 0)    
        
    # Define source position and radius
    source_position = args.source_location
    source_radius = args.source_radius
    feasible_init = args.feasible_init
    
#    print(source_position, source_radius)
#    exit()
    
    # Set discount factor, horizon and number of episodes
    horizon = args.horizon
    num_episodes = args.num_episodes
    gamma = args.gamma
    checkpoint_iters = args.checkpoint_iters
    
    # Set agent velocity
    for i in range(4):
        actions[i] = args.agent_velocity * actions[i]
    
    # Compute seeds
    for key in seeds.keys():
        seeds[key] =(args.base_seed + 1) * seeds[key] + args.base_seed
        
    
    out_dir = f"{args.out_path}/{args.agent_type}_Q/{args.setting}/agent_{seeds['agent_seed']}"
    
    if args.agent_type == 'two':
        out_dir += f"/tau_{args.tau}"
        if args.right_Q == 1:
            out_dir += "/right_Q"
    
    rnd_state = np.random.RandomState(seeds['samp_seed'])
    os.makedirs(f"{out_dir}/Q_stored", exist_ok=True)
    
    # Define the probability to sample initial conditions as uniform in the region of the field
    # where the data is above a certain threshold (the noise threshold of the sparsest environment)
    # and set to 0.0 the probability to sample initial conditions in the source region
    
    if feasible_init == 0 or feasible_init ==2:
        start_points = np.any(data > 7e-6, axis=0).astype(np.float64)
        start_points[:, :int(source_position[1] + source_radius)] = 0.0
        start_points /= np.sum(start_points)
        start_points = start_points.ravel()

    elif feasible_init == 1:

        start_points = (data.copy() > n_thr).astype(np.float64) #np.any(data > n_thr, axis=0).astype(np.float64)

        start_points[:, :, :int(source_position[1] + source_radius)] = 0.0
        start_points = np.argwhere(start_points > 0.0)
    elif feasible_init == 3:
        start_points = np.any(data.copy() > n_thr, axis=0).astype(np.float64) #np.any(data > n_thr, axis=0).astype(np.float64
        start_points[:, :int(source_position[1] + source_radius)] = 0.0
        start_points /= np.sum(start_points)
        start_points = start_points.ravel()

    # Apply the noise threshold to the data   
    data[data < n_thr] = 0.0
    
    sampler_rnd_state = np.random.RandomState(seeds['sim_seed'])

    # Create the environment and the agent
    env = Environment(data, 
                      source_position=source_position, 
                      source_radius=source_radius,
                      gamma = gamma,
                      seed = seeds['env_seed'])
    
    
    def sample_init_position():
        if feasible_init == 0 or feasible_init ==2 or feasible_init ==3:
                
            pos_idx = rnd_state.choice(range(start_points.shape[0]), 1, p=start_points)
            pos_y, pos_x = np.unravel_index(pos_idx, shape=(data.shape[1], data.shape[2]))
            pos = np.concatenate([pos_y, pos_x])
            pos = pos.astype(int)
            if feasible_init == 2:
                feasible_time_slices = data[:, pos[0], pos[1]] > 7e-6
                prob_feasible_time_slices = feasible_time_slices.astype(np.float64) / np.sum(feasible_time_slices)
                time_idx = sampler_rnd_state.choice(len(data[:, pos[0], pos[1]]), 1, p=prob_feasible_time_slices)[0]
            else:
                time_idx = sampler_rnd_state.randint(0, env.MAX_T)
            return pos, time_idx

            
        position_time = start_points[rnd_state.choice(len(start_points))].copy().flatten()

        time_idx = int(position_time[0])
        pos = position_time[1:].astype(int)

        return pos, time_idx

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
    elif args.agent_type == 'two':
        agent = TwoQAgent(
            horizon=horizon,
            gamma=gamma,
            eps_greedy= eps_greedy,
            learning_rate= learning_rate,
            seed=seeds['agent_seed'],
            tau=args.tau,
            use_right_Q=args.right_Q == 1,
            s_thr=n_thr,
            out_dir=f"{out_dir}/Q_stored"
        )
    
    simulator = Simulator(env=env, actions=actions)
    
    # Write the training parameters to a file
    
    write_info_file(out_dir, "training_info", 
                     ["agent_velocity", "agent_type", "setting", "source_position",\
                      "source_radius", "gamma", "horizon", "num_episodes", "eps_init",\
                      "eps_end", "eps_decay", "alpha_init", "alpha_end", "alpha_decay", 
                      "checkpoint_iters", "agent_seed", "env_seed", "samp_seed", "sim_seed"],
                     [args.agent_velocity, args.agent_type, args.setting, source_position, 
                      source_radius, gamma, horizon, num_episodes, eps_init, 
                      eps_end, eps_decay, alpha_init, alpha_end, alpha_decay, 
                      checkpoint_iters, seeds['agent_seed'], seeds['env_seed'], seeds['samp_seed'], 
                      seeds['sim_seed']])


    # Train the agent
    cumulative_rewards, agent_speed, convergence, max_void_steps, num_steps, max_void_lengths, init_positions = simulator.train_agent(agent, 
                                                                        point_sampler=sample_init_position, 
                                                                        horizon=horizon, 
                                                                        num_episodes=num_episodes, 
                                                                        delta=500, 
                                                                        use_right_Q=args.right_Q == 1,
                                                                        checkpoint_iters=checkpoint_iters, 
                                                                        seed=seeds['sim_seed'])
    # Save the Q function and the training log
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
    parser.add_argument("source_location", type=parse_numpy_array, default=np.array([13, 0], dtype=np.int32), help="Source location (y, x)")
    parser.add_argument("source_radius", type=float, default=2.0, help="Source radius")
    
    
    # Agent parameters
    parser.add_argument("--agent_velocity", type=int, default=1, help="Agent velocity")
    parser.add_argument("--agent_type", type=str, choices=['single', 'two'], default='single', help="Type of agent. If 'single' it uses a single Q matrix if 'two' it uses two Q")
    parser.add_argument("--tau", type=int, default=1, help='Threshold for two Q agent')
    parser.add_argument("--right_Q", type=int, choices=[0, 1], default=0, help='For 2Q agent, 1 if choose the Q according to <tb> 0 according to state')

    # Training parameters
    parser.add_argument("--gamma", type=float, default=0.998, help="Discount factor")    
    parser.add_argument("--horizon", type=int, default=500, help="Horizon of the simulation")
    parser.add_argument("--num_episodes", type=int, default=3000001, help="Number of episodes to train the agent")
    parser.add_argument("--eps_init", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--eps_end", type=float, default=0.0001, help="Final epsilon")
    parser.add_argument("--eps_decay", type=float, default=2500000, help="Epsilon decay")
    parser.add_argument("--alpha_init", type=float, default=0.1, help="Initial alpha")
    parser.add_argument("--alpha_end", type=float, default=0.0001, help="Final alpha")
    parser.add_argument("--alpha_decay", type=float, default=2500000, help="Alpha decay")
    parser.add_argument("--checkpoint_iters", type=int, default=10000, help="Checkpoint iterations")
    parser.add_argument("--feasible_init", type=int, choices=[0, 1, 2, 3], default=0, help='1 if initial positions are sampled such that are in the training cone and the first o_t = 1; 2 only if it is in the sparset cone and o_t = 1;  0 if uniformly in the whole grid')

    # Other parameters
    parser.add_argument("--base_seed", type=int, default=143125, help='Seed used by the agent for epsilon-greedy strategy')
    parser.add_argument("--out_path", type=str, default='./results', help='output directory')
    
    args = parser.parse_args()
    train_agent(args)