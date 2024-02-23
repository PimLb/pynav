import numpy as np
from tqdm import tqdm

from pynav.agents import QAgent
from pynav.environment import Environment


class Simulator:
    
    def __init__(self, env : Environment, actions) -> None:
        self.env = env
        self.actions = actions
        
    def _perform_single_step(self, agent : QAgent, pos : np.ndarray, time_idx : int, deterministic : bool = False):
        a = agent.select_action(deterministic=deterministic)
        pos, r, terminated = self.env.make_move(pos, self.actions[a])
        time_idx = (time_idx + 1) % self.env.MAX_T
        obs = self.env.get_observation(pos, time_idx)
        s_prime = agent.update_state(obs)
        return pos, a, r, terminated, s_prime, time_idx

        
    def train_agent(self, 
                    agent : QAgent, # agent to train
                    point_sampler, # probability of observation at every location 
                    horizon, # number of steps per episode (horizon)
                    num_episodes, # number of episodes
                    delta : int = 500, # size of the window used to compute the average cumulative reward per episode
                    seed : int = 12131415 # random seed
                    ):
        cumulative_rewards = []
        average_crewards = []
        rnd_state = np.random.RandomState(seed=seed)

        iterator = tqdm(range(num_episodes))
        for _ in iterator:
            # Initialization
            agent.reset()
            pos = point_sampler()
            if pos[1] <= self.env.source_position[1] + self.env.source_radius or self.env.source_reached(pos):
                continue
            time_idx = rnd_state.randint(0, self.env.MAX_T)
            obs = self.env.get_observation(pos, time_idx)
            s = agent.update_state(obs) # update memory and compute olfactory state
            c_reward = 0.0
            for t in range(horizon):
                pos, a, r, terminated, s_prime, time_idx = self._perform_single_step(agent, 
                                                                                  pos, 
                                                                                  time_idx, 
                                                                                  deterministic = False)

                c_reward += r * (agent.gamma**t) if t > 0 else r
                agent.update_q(s, a, s_prime, r)
                if terminated:
                    break
                s = s_prime
            agent.num_episodes += 1
            cumulative_rewards.append(c_reward)
            iterator.set_postfix({'avg R_t' : np.mean(cumulative_rewards[-delta:]), 
                                'eps' : f'{agent.eps_greedy(agent.num_episodes)}',
                                'alpha' : f'{agent.learning_rate(agent.num_episodes)}'})
            average_crewards.append(np.mean(cumulative_rewards[-delta:]))
        return cumulative_rewards, average_crewards

    def test_agent(self, agent : QAgent, test_points, time_slices, horizon : int):
        num_converged = 0
        num_valid_pos = 0
        T_ratio_lists = []
        failed_positions = []
        for time_slice in time_slices:
            T_ratio = []
            iterator = tqdm(range(test_points.shape[0]))
            for pos_idx in iterator:
                init_pos = test_points[pos_idx]
                
                if init_pos[1] <= self.env.source_position[1] + self.env.source_radius or self.env.source_reached(init_pos):
                    continue
                num_valid_pos += 1
                agent.reset()
                time_idx = time_slice
                obs = self.env.get_observation(init_pos, time_idx)
                s = agent.update_state(obs) # update memory and compute olfactory state
                trace = []
                source_pos = np.array([14.0, 0.0])
                c_reward = 0.0
                pos = init_pos.copy()
                for t in range(horizon):
                    pos, a, r, terminated, s_prime, time_idx = self._perform_single_step(agent, 
                                                                                  pos, 
                                                                                  time_idx, 
                                                                                  deterministic = True)
                    c_reward += r * (agent.gamma**t) if t > 0 else r
                    trace.append((s, a, s_prime, r))
                    if terminated:
                        T_min = np.abs(source_pos[0] - init_pos[0]) + np.abs(source_pos[1] - init_pos[1]) - 2
                        iterator.set_postfix({'t':time_slice,f"Converged! steps" : f"{t} steps", "T_min / T " : T_min / (t + 1)})
                        T_ratio.append(T_min / (t + 1))
                        num_converged +=1
                        break
                    s = s_prime
                if terminated:
                    failed_positions.append(init_pos.copy())
            T_ratio_lists.append(T_ratio)
            return T_ratio_lists
