import numpy as np


class StepSize:
    
    def __init__(self, mode : str, init_value : float, decay : float = 1e-2, min_val = None) -> None:
        assert mode in ['constant', 'linear', 'exponential', 'sqrt']
        self.update = getattr(StepSize, "_" + mode)
        self.init_value = init_value
        self.alpha = init_value
        self.mode = mode
        self.decay = decay
        self.min_val = min_val
        self.num_episode = 0

    def _constant(self) -> float:
        return self.init_value
    
    def _linear(self) -> float:
        return self.init_value / (1.0 + (self.decay * self.num_episode))
    
    def _exponential(self) -> float:
        return self.init_value * np.exp(-self.decay * self.num_episode)
    
    def _sqrt(self) -> float:
        return self.init_value / (1 + self.decay *np.sqrt(self.num_episode))
        
    def __call__(self) -> float:
        if self.min_val is not None:
            return max(self.alpha, self.min_val)
        return self.alpha

    def next(self) -> None:
        self.num_episode += 1
        self.alpha = self.update(self)   

    def __repr__(self) -> str:
        return "init: {}\tcurrent: {}\tmode: {}\tdecay: {}\tepisodes: {}".format(self.init_value, self.alpha, self.mode, self.decay, self.num_episode)