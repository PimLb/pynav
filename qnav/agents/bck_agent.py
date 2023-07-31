import numpy as np

from qnav.agents.agent import Agent
from qnav.fmap.fmap import FMap
from qnav.environment.env import Environment
from qnav.policy.policy import ImplicitPolicy
from qnav.qfunctions.qfunctions import TabularQ
from qnav.stepsize.stepsize import StepSize


class BacktrackingAgent(Agent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backtracking_stack = []
        self.backtrack = False
        
        
    def backtracking_action(self, state, features, s_thr, test : bool = False):
        if features['avg_int'] > 0.0:
            self.backtracking_stack = []
            self.backtrack = False
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif len(self.backtracking_stack) == 0:
            return self.rnd_state.randint(0, len(self.actions.keys()))             
        return self.backtracking_stack.pop()         


        
    def get_action(self, state, features, s_thr, test : bool = False, action = None):
        if action is None:
            # if this is the first move, play as sensing agent
            memory_is_empty = len(self.memory[self.memory >= s_thr]) == 0
            if test and self.rnd_on_zero and memory_is_empty:
                return self.rnd_state.randint(0, len(self.actions.keys()))
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        if self.backtrack:
            return self.backtracking_action(state, features, s_thr, test)
        last_is_zero = self.memory[-1] < s_thr
        if features['avg_int'] > 0.0 and not last_is_zero:
            # if non-zero is observed, reset the backtrack stack and 
            # play action according to Q-learning policy
            self.backtracking_stack = []
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif features['avg_int'] > 0.0 and last_is_zero:
            # if zero is observed and average intensity in memory is not zero,  
            # add the inverse of the action played in the backtracking stack
            self.backtracking_stack.append(self.inverse_actions[action])
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif features['avg_int'] == 0.0:
            # if memory is empty, then
            # if backtrack stack is not empty => play the inverse action
            if len(self.backtracking_stack) > 0:
                self.backtrack = True
                return self.inverse_actions[action]
            # if backtrack stack is empty => play a random action
            return self.rnd_state.randint(0, len(self.actions.keys()))
            
               
                
    def reset(self):
        super().reset()
        self.backtracking_stack = []
        self.backtrack = False
