import sys 
import os
import argparse as ap
import numpy as np 
import matplotlib.pyplot as plt 
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


horizon = 500 #1000

gamma = 0.998 # Because the tipical path len should be ~ 500

source_position = np.array([13, 0])
source_radius = 2



checkpoint_iters = 10000

def plot_positions(data, position_lists, name, out_dir, Q_used = None):
    fig, ax =plt.subplots(figsize=(12, 6))
    
    
    ax.imshow(np.log10(data[0, :, :]), origin='lower')
    
    for i in range(len(position_lists)):
        pos = position_lists[i]
        if Q_used is None:
            ax.plot(pos[1], pos[0], 'o', c='black')
        else:
            color = 'red' if Q_used[i] == 0 else 'orange'
            ax.plot(pos[1], pos[0], 'o', c=color)
    
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches='tight')
    plt.close(fig)


def test_agent(args):
    n_thr = settings[args.test_setting]
    data = np.load(args.data_path)
    
    
    agent_dir = f"{args.agent_path}/two_Q/{args.training_setting}/agent_{args.agent_seed}"
    
    agent_dir += f"/tau_{args.tau}"
    
    if args.q_num is None:
        out_dir = f"{agent_dir}/test/{args.test_setting}/only_Q_{args.qtype}"
    else:
        out_dir = f"{agent_dir}/test/{args.test_setting}/Q_{args.q_num}/only_Q_{args.qtype}"
    
    
    os.makedirs(f"{out_dir}", exist_ok=True)
    
    start_points = np.any(data > 7e-6, axis=0).astype(np.float64)
    start_points[:, :3] = 0.0 # Set to 0.0 the probability to sample initial conditions in the source region
    
    data[data < n_thr] = 0.0
    env = Environment(data, 
                      source_position=source_position, 
                      source_radius=source_radius,
                      gamma = gamma,
                      seed = args.agent_seed)


    
    agent = QAgent(
        horizon=horizon,
        gamma=gamma,
        seed=args.agent_seed,
        s_thr=n_thr,
        out_dir=f"{agent_dir}/Q_stored"
    )

        
    if args.q_num is None:
        Q = np.load(f"{agent_dir}/Q_stored/Qfun.npy", allow_pickle=True)
    elif args.q_num > 0:
        Q = np.load(f"{agent_dir}/Q_stored/Qfun_{args.q_num}.npy", allow_pickle=True)

    if args.qtype == 0:
        Q = Q[:horizon + 3, :]
    else:
        Q = np.vstack((Q[0, :], Q[horizon + 3:, :]))
    agent.Q = Q
    
    simulator = Simulator(env=env, actions=actions)
    
    test_points = np.transpose(np.nonzero(start_points))#.T

    results, failed_positions, last_detections, Q_last_detections, Q_used, Q_used_list = simulator.test_agent(agent, 
                         test_points=test_points,
                         time_slices=[0, 100, 500, 1000],#range(data.shape[0]),
                         horizon=horizon,
                         get_Q_used_map=False
                         )
    data[data == 0.0] = 1e-20
    
    plot_positions(data, failed_positions, "failed_init_positions", f"{out_dir}")
    plot_positions(data, last_detections, "failed_last_detections", f"{out_dir}", Q_used=Q_last_detections)

    with open(f"{out_dir}/failed_init_positions.log" , 'w') as f:
        for i in range(len(failed_positions)):
            f.write("{},{}\n".format(failed_positions[i][0], failed_positions[i][1]))

    with open(f"{out_dir}/Q_used_list.log" , 'w') as f:
        for i in range(len(Q_used_list)):
            hist, bin_edges = np.histogram(Q_used_list[i], bins=2)
            f.write("{},{},{}\n".format(i,hist[0], hist[1]))
            f.flush()


    with open(f"{out_dir}/failed_last_detections.log" , 'w') as f:
        for i in range(len(last_detections)):
            f.write("{},{},{}\n".format(last_detections[i][0], last_detections[i][1], Q_last_detections[i]))

    creward, speed, convergence = [],[], []

    with open(f"{out_dir}/test.log", 'w') as f:
        #(init_pos[0], init_pos[1], time_slice, c_reward, speed, int(terminated))
        for result in results:
            creward.append(result[3])
            if result[5]:
                speed.append(result[4])
            convergence.append(result[5])
            f.write(",".join([str(elem) for elem in result]) + "\n")
            
    print("[--] G = {} +/- {}".format(np.mean(creward), np.std(creward)))
    print("[--] speed = {} +/- {}".format(np.mean(speed), np.std(speed)))
    print("[--] convergence = {} +/- {}".format(np.mean(convergence), np.std(convergence)))
    with open(f"{out_dir}/test_summary.log", 'w') as f:
        mu_crew, std_crew = np.mean(creward) if len(creward) > 0 else 0.0, np.std(creward) if len(creward) > 0 else 0.0
        mu_speed, std_speed = np.mean(speed) if len(speed) > 0 else 0.0, np.std(speed) if len(speed) > 0 else 0.0
        mu_conv, std_conv = np.mean(convergence) if len(convergence) > 0 else 0.0, np.std(convergence) if len(convergence) > 0 else 0.0
        f.write(f"{mu_crew},{std_crew},{mu_speed},{std_speed},{mu_conv},{std_conv}\n")
    
    
    if Q_used is not None:
        with open(f"{out_dir}/Q_used_map.log", 'w') as f:
            Q_used_map = np.full((data.shape[1], data.shape[2]), -1.0)
            for (pos, values) in Q_used.items():
                array_pos = np.fromstring(pos.replace("[", "").replace("]", ""), sep=" ", dtype=int)
                Q_used_map[array_pos[0], array_pos[1]] = np.mean(values)
                f.write("{},{},{},{}\n".format(array_pos[0], array_pos[1], np.mean(values), np.std(values)))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(Q_used_map, origin='lower')
        fig.savefig(f"{out_dir}/Q_used.pdf", bbox_inches='tight')
        plt.close(fig)
#            f.write(f"{elem},{x},{t},{c_rew},{speed},{terminated}\n")
            


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Test QAgent.", formatter_class=ap.ArgumentDefaultsHelpFormatter)

    # Data parameters
    parser.add_argument("data_path", type=str, help="path to ,file containing data")
    parser.add_argument("training_setting", type=str, choices=['dense', 'middle', 'sparse', 'sparser'], help='setting used to train the agent')
    parser.add_argument("test_setting", type=str, choices=['dense', 'middle', 'sparse', 'sparser'], help='setting used to test the agent')
    parser.add_argument("--agent_seed", type=int, default=143125, help='Seed used by the agent for epsilon-greedy strategy')
    parser.add_argument("--agent_path", type=str, default='./results', help='output directory')


    # Experiment parameters
    parser.add_argument("--qtype", type=int, choices=[0, 1], default=0, help="Type of Qagent. ")
    parser.add_argument("--tau", type=int, default=1, help="Threshold for switching Q in TwoQAgent")
    
    parser.add_argument('--q_num', type=int, default=None, help='Num of Q')
    
    args = parser.parse_args()
    test_agent(args)