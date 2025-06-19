import numpy as np 
import argparse as ap


def parse_numpy_array(string):
    try:
        return np.fromstring(string, sep=',')
    except Exception as e:
        raise ap.ArgumentTypeError(f"Invalid array input: {string}")

def write_info_file(out_dir, name, labels, values):
    with open(f"{out_dir}/{name}.txt", "w") as f:
        for key, value in zip(labels, values):
            f.write(f"[--] {key}: {value}\n")