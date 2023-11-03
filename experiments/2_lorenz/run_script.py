import pathlib
import sys
import os
import subprocess
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)

from generate_data import DATA_DIR, ScaleTsfm

from tqdm.contrib.concurrent import process_map

def run_train_ctvae(tf_seed_dpath):
    tf, seed, dpath = tf_seed_dpath
    command = f"python train_ctvae.py -rs {seed} --dpath {dpath} -tf {tf}"
    subprocess.run(command.split(" "), cwd=CURR_DIR, check=True)

def main():
    dpath_list = [dpath for dpath in os.listdir(DATA_DIR) if dpath.endswith(".pkl")]
    dpath_list = [os.path.join(DATA_DIR, dpath) for dpath in dpath_list]
    experiments = []
    for dpath in dpath_list:
        seed = int(dpath.split("_")[-1].split(".")[0])
        tf = float(dpath.split("_")[-2])
        experiments.append((tf, seed, dpath))

    # training the CTVAEs
    process_map(run_train_ctvae, experiments, max_workers=10)

    # getting gradients from adjoints
    subprocess.run("python train_odeint.py", shell=True, check=True)
    
if __name__ == "__main__":
    main()
