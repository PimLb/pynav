import numpy as np

def estimate_tb_duration(data, s_thr):
    indices = []
    tb_duration = []
    for i in range(len(data)):
        if data[i] >= s_thr:
            indices.append(i)
    for i in range(len(indices) - 1):
        delta = abs(indices[i] - indices[i + 1]) - 1
        if delta > 0:
            tb_duration.append(delta)
    return tb_duration, indices