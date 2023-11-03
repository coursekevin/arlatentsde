import os
import random
import pathlib
import subprocess

from tqdm.contrib.concurrent import process_map

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
CKPT_DIR = os.path.join(CURR_DIR, "ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)


def run_experiment(rs_batch_size: tuple[int, int]):
    rs, batch_size = rs_batch_size
    if not experiment_complete(rs, batch_size):
        command = f"python train_ctvae.py -rs {rs} -S {batch_size}"
        subprocess.run(command, shell=True, check=True)
    else:
        print(f"Experiment {batch_size} | {rs} already complete.")


def is_matching_int(dir: str, num: int) -> bool:
    return num == int(dir.split("_")[1])


def experiment_complete(rs, batch_size) -> bool:
    ckpt_dirs = [
        dir for dir in os.listdir(CKPT_DIR) if is_matching_int(dir, batch_size)
    ]
    for dir in ckpt_dirs:
        exp_dirs = os.listdir(os.path.join(CKPT_DIR, dir))
        if any(is_matching_int(exp, rs) for exp in exp_dirs):
            return True
    return False


if __name__ == "__main__":
    rs_range = range(10)
    mc_range = [10, 50, 100]
    exp_args = [(rs, mc) for mc in mc_range for rs in rs_range]
    random.shuffle(exp_args)
    process_map(run_experiment, exp_args, max_workers=3)

