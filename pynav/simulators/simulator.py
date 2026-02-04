import numpy as np
from tqdm import tqdm

from pynav.agents import QAgent, TwoQAgent
from pynav.environment import Environment




class Simulator:
    
    def __init__(self, env : Environment, actions) -> None:
        self.env = env
        self.actions = actions
        
    def _perform_single_step(self, agent : QAgent, pos : np.ndarray, time_idx : int, deterministic : bool = False, use_right_Q : bool = False):
        a = agent.select_action(deterministic=deterministic)
        pos, r, terminated = self.env.make_move(pos, self.actions[a % 4])
        time_idx = (time_idx + 1) % self.env.MAX_T
        if use_right_Q and isinstance(agent, TwoQAgent):
            agent.Q_used = int(self.env.compute_tb(pos, agent.s_thr) > agent.tau)

        obs = self.env.get_observation(pos, time_idx) if not terminated else -1.0
        s_prime = agent.update_state(obs)
        return pos, a, r, terminated, s_prime, time_idx, obs

        
    def train_agent(self, 
                    agent : QAgent, # agent to train
                    point_sampler, # probability of observation at every location 
                    horizon, # number of steps per episode (horizon)
                    num_episodes, # number of episodes
                    delta : int = 500, # size of the window used to compute the average cumulative reward per episode
                    seed : int = 12131415, # random seed
                    use_right_Q : bool = False, # use right Q function in TwoQAgent
                    checkpoint_iters : int = 1000
                    ):
        cumulative_rewards, agent_speed, convergence, max_void_steps, num_steps = [], [], [], [], []
        init_positions =[]
        rnd_state = np.random.RandomState(seed=seed)

        max_void_length = 0 
        max_void_lengths = np.zeros(num_episodes)

        iterator = tqdm(range(num_episodes))
        for k in iterator:
            # Initialization
            agent.reset()
            pos, time_idx = point_sampler()

            if pos[1] <= self.env.source_position[1] + self.env.source_radius or self.env.source_reached(pos):
                continue
            init_positions.append(pos.copy())
            init_pos = pos.copy()
#            time_idx = rnd_state.randint(0, self.env.MAX_T)
            if use_right_Q and isinstance(agent, TwoQAgent):
                agent.Q_used = int(self.env.compute_tb(pos, agent.s_thr) > agent.tau)
                
            obs = self.env.get_observation(pos, time_idx)
            s = agent.update_state(obs) 

            c_reward = 0.0
            previous_blank = agent.current_blank
            max_void_length = agent.current_blank
            void_steps = agent.current_blank
            for t in range(horizon):
                pos, a, r, terminated, s_prime, time_idx, _ = self._perform_single_step(agent, 
                                                                                  pos, 
                                                                                  time_idx, 
                                                                                  deterministic = False,
                                                                                  use_right_Q = use_right_Q)
                
                c_reward += r * (agent.gamma**t) 
                agent.update_q(s, a, s_prime, r)
                
                void_steps = max(void_steps, agent.current_blank)
                if s_prime == 0:
                    max_void_length = max(max_void_length, previous_blank)

                if terminated:
                    break
                s = s_prime
                previous_blank = agent.current_blank
            max_void_steps.append(void_steps)
            num_steps.append(t + 1)
            convergence.append(int(terminated))
            episode_speed = 0.0 if not terminated else self.env.get_distance_to_source(init_pos) / (t + 1)
            agent_speed.append(episode_speed)
            if ((k + 1) % checkpoint_iters) == 0:
                agent.store_q(f"Qfun_{k + 1}")
            max_void_lengths[k] = max_void_length
            cumulative_rewards.append(c_reward)
            iterator.set_postfix({'init position' : init_positions[-1],
                                  'avg G_t' : np.mean(cumulative_rewards[-delta:]), 
                                  'avg T_min/T_t' : np.mean(agent_speed[-delta:]),
                                  'avg C_t' : np.mean(convergence[-delta:]),
                                  'max void steps' : np.mean(max_void_steps[-delta:]),
                                  'eps' : agent.eps_greedy(agent.num_episodes),
                                  'alpha' : agent.learning_rate(agent.num_episodes)})
            agent.num_episodes += 1
        return cumulative_rewards, agent_speed, convergence, max_void_steps, num_steps, max_void_lengths, init_positions

    def test_agent(self, agent : QAgent, test_points, time_slices, horizon : int, get_Q_used_map = False, init_with_tb = False , feasible_init = False):
        failed_positions = []
        last_detections = []
        Q_last_detections = []
        results = []
        Q_used = {}
        Q_used_list = [[] for _ in range(horizon + 1)]
        for time_slice in time_slices:
            iterator = tqdm(range(test_points.shape[0]))
            for pos_idx in iterator:
                init_pos = test_points[pos_idx]
                                    
                if init_pos[1] <= self.env.source_position[1] + self.env.source_radius or self.env.source_reached(init_pos):
                    continue
                speed = 0.0
                agent.reset()
                if init_with_tb:
                    tb = self.env.data[:, init_pos[0], init_pos[1]] > agent.s_thr
                    count_zero = 0 
                    avg_blank = []
                    for k in range(0, len(tb)):
                        if not tb[k]:
                            count_zero +=1
                        else:
                            if count_zero > 0:
                                avg_blank.append(count_zero)
                            count_zero = 0
                    if count_zero > 0:
                        avg_blank.append(count_zero)
                    agent.Q_used = int(np.mean(avg_blank) > agent.tau) #if len(avg_blank) > 0 else 1

                time_idx = time_slice
                obs = self.env.get_observation(init_pos, time_idx)
                s = agent.update_state(obs) # update memory and compute olfactory state
#                trace = []
#                source_pos = np.array([14.0, 0.0])
                c_reward = 0.0
                pos = init_pos.copy()
                has_observed = obs > 0.0 
                last_detection = None if s > 0.0 else init_pos.copy()
                Q_last_detection = None if s > 0.0 else agent.Q_used #init_pos.copy()
                max_void_length = agent.current_blank
                if get_Q_used_map:
                    if str(pos) not in Q_used:
                        Q_used[str(pos)] = [agent.Q_used]
                    else:
                        Q_used[str(pos)].append(agent.Q_used)
                Q_used_list[0].append(agent.Q_used)
                for t in range(horizon):
                    pos, a, r, terminated, s_prime, time_idx, obs = self._perform_single_step(agent, 
                                                                                  pos, 
                                                                                  time_idx, 
                                                                                  deterministic = True)
                    Q_used_list[t + 1].append(agent.Q_used)
                    c_reward += r * (agent.gamma**t) #if t > 0 else r
                    has_observed = has_observed or (obs > 0.0)
                    if get_Q_used_map and 0 <= pos[0] < self.env.data.shape[1] and 0 <= pos[1] < self.env.data.shape[2]:
                        if str(pos) not in Q_used:
                            Q_used[str(pos)] = [agent.Q_used]
                        else:
                            Q_used[str(pos)].append(agent.Q_used)
                    if has_observed and s_prime == 0.0:
                        print("[--] observation at: ", pos, "time idx:", time_idx, obs)
                        last_detection = pos.copy()
                        Q_last_detection = agent.Q_used
                        max_void_length = max(max_void_length, agent.current_blank)
                    if terminated:
                        T_min =  self.env. get_distance_to_source(init_pos) #np.abs(source_pos[0] - init_pos[0]) + np.abs(source_pos[1] - init_pos[1]) - 2
                        iterator.set_postfix({'t':time_slice, "pos" : init_pos, f"Converged! steps" : f"{t} steps", "T_min / T " : T_min / (t + 1)})
                        speed = T_min / (t + 1)
                        break
                    s = s_prime
                print("-----------------------------")
                if not terminated:
                    failed_positions.append(init_pos.copy())
                    speed = 0.0
                    if last_detection is not None:
                        last_detections.append(last_detection)
                        Q_last_detections.append(Q_last_detection)
                results.append([init_pos[0], init_pos[1], time_slice, c_reward, speed, int(terminated), max_void_length])
        if not get_Q_used_map:
            Q_used = None
        return results, failed_positions, last_detections, Q_last_detections, Q_used, Q_used_list

