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
    
    def __init__(self, n_thr, thresholds):
        self.n_thr = n_thr
        self.intensities = np.asarray([self.n_thr])
        super().__init__(thresholds)
        self._init_avg_threshold()
        
    def _init_avg_threshold(self):
        self.thresholds[1][0] = self.intensities[0] / 4
        self.thresholds[1][1] = self.intensities[0] / 2
        self.thresholds[1][2] = self.intensities[0] * 2
        self.thresholds[1][3] = self.intensities[0] * 4
    
    def _update_int_estimation(self):
#        if len(self.intensities) > 0:
        p1, p2, p3, p4 = np.percentile(self.intensities, 25), np.percentile(self.intensities, 50), np.percentile(self.intensities, 80), np.percentile(self.intensities, 99)
        self.thresholds[1][0] = p1 if p1 > 0.0 else self.thresholds[1][0]
        self.thresholds[1][1] = p2 if p2 > p1 else self.thresholds[1][1]
        self.thresholds[1][2] = p3 if p3 > p2 else self.thresholds[1][2]
        self.thresholds[1][3] = p4 if p4 > p3 else self.thresholds[1][3]
        
    def to_index(self, phi):
        state_idx = super().to_index(phi)
        return state_idx
        
    def update(self, raw, features):
        intensity = features['avg_int']
        self.intensities = np.concatenate([self.intensities, [intensity]])
        self._update_int_estimation()
    
    def reset(self):
        self.intensities = np.array([self.n_thr])
        self.thresholds[1] = np.zeros(4)
        self._init_avg_threshold()



