import numpy as np
from pynav.feature_maps.fmap import FeatureMap

interm_disc = [0.33, 0.66]

fmap = FeatureMap(interm_disc=interm_disc)

fmap.avgint_disc[0] = 1.0
fmap.avgint_disc[1] = 2.0
fmap.avgint_disc[2] = 3.0
fmap.avgint_disc[3] = 4.0

for a_int in [1.0, 1.5, 2.5, 3.5, 4.0]:
    for o_int in [0.0, 0.1, 0.3, 0.33, 0.4, 0.5, 0.55, 0.66, 0.7, 0.8, 1.0]:
        idx = fmap(dict(average_intensity=a_int, intermittency=o_int))
        print("[--] [{}, {}] => {}".format(o_int, a_int, idx))
