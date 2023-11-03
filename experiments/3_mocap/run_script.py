import subprocess
import pathlib

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())


def main():
    random_seeds = [i for i in range(10)]
    for seed in random_seeds:
        subprocess.run(
            ["python", "train_ctvae.py", "-rs", str(seed)], cwd=CURR_DIR, check=True
        )

if __name__ == "__main__":
    main()
