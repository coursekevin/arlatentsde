import subprocess

if __name__ == "__main__":
    rs_range = range(10)
    for rs in rs_range:
        command = f"python train_ctvae.py -rs {rs}"
        subprocess.run(command, shell=True, check=True)

    tol_range = [1e-2, 1e-4, 1e-6]
    for tol in tol_range:
        for rs in rs_range:
            command = f"python train_odeint.py -rs {rs} -tol {tol}"
            subprocess.run(command, shell=True, check=True)
