import numpy as np

class FixedDiscretization:
    
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
        idx = 0
        disc_phi = np.zeros(len(phi), dtype=np.int32)
        for i in range(len(disc_phi)):
            disc_phi[i] = self.discretize(phi[i], i)
        return tuple(disc_phi)



class FMap(FixedDiscretization):
    
    def __init__(self, thresholds):
        self.intensities = []
        self.intensities = np.asarray([0.0])
        super().__init__(thresholds)
    
    def _update_int_estimation(self, o_thr):
        if len(self.intensities) > 0:
            self.thresholds[1][0] = np.percentile(self.intensities, 25)
            self.thresholds[1][1] = np.percentile(self.intensities, 50)
            self.thresholds[1][2] = np.percentile(self.intensities, 80)
            self.thresholds[1][3] = np.percentile(self.intensities, 99)
        
    def to_index(self, phi):
        state_idx = super().to_index(phi)
        return state_idx
        
    def update(self, raw, features, o_thr):
        intensity = features['avg_int']
        self.intensities = np.concatenate([self.intensities, [intensity]])
        self._update_int_estimation(o_thr)
    
    def reset(self):
        self.intensities = [0.0]
        self.thresholds[1] = np.zeros(4)
