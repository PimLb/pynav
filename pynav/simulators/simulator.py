import numpy as np
from tqdm import tqdm

from pynav.agents import QAgent, TwoQAgent
from pynav.environment import Environment




class Simulator:
    
    def __init__(self, env : Environment, actions) -> None:
        self.env = env
        self.actions = actions
        
    def _perform_single_step(self, agent : QAgent, pos : np.ndarray, time_idx : int, deterministic : bool = False):
        a = agent.select_action(deterministic=deterministic)
        pos, r, terminated = self.env.make_move(pos, self.actions[a])
        time_idx = (time_idx + 1) % self.env.MAX_T
        obs = self.env.get_observation(pos, time_idx) if not terminated else -1.0
        s_prime = agent.update_state(obs)
        return pos, a, r, terminated, s_prime, time_idx

        
    def train_agent(self, 
                    agent : QAgent, # agent to train
                    point_sampler, # probability of observation at every location 
                    horizon, # number of steps per episode (horizon)
                    num_episodes, # number of episodes
                    delta : int = 500, # size of the window used to compute the average cumulative reward per episode
                    seed : int = 12131415, # random seed
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
            pos = point_sampler()
            
 #           if isinstance(agent, TwoQAgent):
                # blanks = []
                # non_voids = self.env.data[:, pos[0], pos[1]] > agent.s_thr
                # init_idx = np.argmax(non_voids)
                # b_size = 0
                # blank_open = False
                # for i in range(init_idx, self.env.data.shape[0]):
                #     if non_voids[i] and not blank_open:
                #         blank_open = True
                #     elif non_voids[i] and blank_open:
                #         if b_size > 0:
                #             blanks.append(b_size)
                #         b_size = 0
                #     elif not non_voids[i]:
                #         b_size +=1
#                agent.Q_used = int(np.mean(blanks) > agent.tau)
#                agent.current_blank = np.mean(blanks)
#                agent.current_state = horizon + 2 if agent.Q_used else 0
                    
                    

            if pos[1] <= self.env.source_position[1] + self.env.source_radius or self.env.source_reached(pos):
                continue
            init_positions.append(pos.copy())
            init_pos = pos.copy()
            time_idx = rnd_state.randint(0, self.env.MAX_T)
            obs = self.env.get_observation(pos, time_idx)
            s = agent.update_state(obs) 

            c_reward = 0.0
            previous_blank = agent.current_blank
            max_void_length = agent.current_blank
            void_steps = agent.current_blank
            for t in range(horizon):
                pos, a, r, terminated, s_prime, time_idx = self._perform_single_step(agent, 
                                                                                  pos, 
                                                                                  time_idx, 
                                                                                  deterministic = False)

                c_reward += r * (agent.gamma**t) 
                agent.update_q(s, a, s_prime, r)
                
#                print(agent.Q_used)
                
                void_steps = max(void_steps, agent.current_blank)
                if s_prime == 0:
                    max_void_length = max(max_void_length, previous_blank)

                if terminated:
                    break
                s = s_prime
                previous_blank = agent.current_blank
#            print("-"*55)
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

    def test_agent(self, agent : QAgent, test_points, time_slices, horizon : int, get_Q_used_map = False):
        failed_positions = []
        last_detections = []
        Q_last_detections = []
        results = []
        Q_used = {}
        for time_slice in time_slices:
            iterator = tqdm(range(test_points.shape[0]))
            for pos_idx in iterator:
                init_pos = test_points[pos_idx]
                
                if init_pos[1] <= self.env.source_position[1] + self.env.source_radius or self.env.source_reached(init_pos):
                    continue
                speed = 0.0
                agent.reset()
                time_idx = time_slice
                obs = self.env.get_observation(init_pos, time_idx)
                s = agent.update_state(obs) # update memory and compute olfactory state
#                trace = []
#                source_pos = np.array([14.0, 0.0])
                c_reward = 0.0
                pos = init_pos.copy()
                last_detection = None if s > 0.0 else init_pos.copy()
                Q_last_detection = None if s > 0.0 else agent.Q_used #init_pos.copy()
                max_void_length = agent.current_blank
                if get_Q_used_map:
                    if str(pos) not in Q_used:
                        Q_used[str(pos)] = [agent.Q_used]
                    else:
                        Q_used[str(pos)].append(agent.Q_used)
                for t in range(horizon):
                    pos, a, r, terminated, s_prime, time_idx = self._perform_single_step(agent, 
                                                                                  pos, 
                                                                                  time_idx, 
                                                                                  deterministic = True)
                    c_reward += r * (agent.gamma**t) #if t > 0 else r
                    if get_Q_used_map and 0 <= pos[0] < self.env.data.shape[1] and 0 <= pos[1] < self.env.data.shape[2]:
                        if str(pos) not in Q_used:
                            Q_used[str(pos)] = [agent.Q_used]
                        else:
                            Q_used[str(pos)].append(agent.Q_used)
                    if s_prime == 0.0:
                        last_detection = pos.copy()
                        Q_last_detection = agent.Q_used
                        max_void_length = max(max_void_length, agent.current_blank)
                    if terminated:
                        T_min =  self.env. get_distance_to_source(init_pos) #np.abs(source_pos[0] - init_pos[0]) + np.abs(source_pos[1] - init_pos[1]) - 2
                        iterator.set_postfix({'t':time_slice, "pos" : init_pos, f"Converged! steps" : f"{t} steps", "T_min / T " : T_min / (t + 1)})
                        speed = T_min / (t + 1)
                        break
                    s = s_prime
                if not terminated:
                    failed_positions.append(init_pos.copy())
                    speed = 0.0
                    if last_detection is not None:
                        last_detections.append(last_detection)
                        Q_last_detections.append(Q_last_detection)
                results.append([init_pos[0], init_pos[1], time_slice, c_reward, speed, int(terminated), max_void_length])
        if not get_Q_used_map:
            Q_used = None
        return results, failed_positions, last_detections, Q_last_detections, Q_used
