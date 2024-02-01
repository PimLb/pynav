import numpy as np


class FeatureMap:
    
    def __init__(self, thresholds) -> None:        
        self.thresholds = thresholds

    def discretize(self, phi_i, i):
        if phi_i <= self.thresholds[i][0]:
            return 0
        if phi_i > self.thresholds[i][-1]:
            return len(self.thresholds[i]) 
        for j in range(1, len(self.thresholds[i])):
            if self.thresholds[i][j - 1] < phi_i <= self.thresholds[i][j]:
                return j
        
    def to_index(self, phi):
        disc_phi = np.zeros(len(phi), dtype=np.int32)
        for i in range(len(disc_phi)):
            disc_phi[i] = self.discretize(phi[i], i)
        return tuple(disc_phi)

