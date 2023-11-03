import argparse
import os
import subprocess
import pathlib
from typing import Protocol
from enum import Enum

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())


class Actions(Enum):
    get_data = "get-data"
    train = "train"
    post_process = "post-process"


class Experiments(Enum):
    predprey = "predprey"
    lorenz = "lorenz"
    mocap = "mocap"
    nsde_video = "nsde-video"
    grad_variance = "grad-variance"


class ActionHandler(Protocol):
    def get_data(self):
        ...

    def train(self):
        ...

    def post_process(self):
        ...


class PredPreyActionHandler:
    def __init__(self):
        self.base_dir = os.path.join(CURR_DIR, "1_predprey")

    def get_data(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "generate_data.py")],
            cwd=self.base_dir,
        )

    def train(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "run_script.py")], cwd=self.base_dir
        )

    def post_process(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "post_process.py")],
            cwd=self.base_dir,
        )


class LorenzActionHandler:
    def __init__(self):
        self.base_dir = os.path.join(CURR_DIR, "2_lorenz")

    def get_data(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "generate_data.py")],
            cwd=self.base_dir,
        )

    def train(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "run_script.py")], cwd=self.base_dir
        )

    def post_process(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "post_process.py")],
            cwd=self.base_dir,
        )


class MocapActionHandler:
    def __init__(self) -> None:
        self.base_dir = os.path.join(CURR_DIR, "3_mocap")

    def get_data(self):
        print("Data is automatically downloaded in training.")

    def train(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "run_script.py")], cwd=self.base_dir
        )

    def post_process(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "post_process.py")],
            cwd=self.base_dir,
        )


class NSDEVideoActionHandler:
    def __init__(self) -> None:
        self.base_dir = os.path.join(CURR_DIR, "4_neural_sde_video")

    def get_data(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "generate_data.py")],
            cwd=self.base_dir,
        )

    def train(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "train_ctvae.py")], cwd=self.base_dir
        )

    def post_process(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "post_process.py")],
            cwd=self.base_dir,
        )


class GradVarianceActionHandler:
    def __init__(self) -> None:
        self.base_dir = os.path.join(CURR_DIR, "5_gradient_variance")

    def get_data(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "generate_data.py")],
            cwd=self.base_dir,
        )

    def train(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "run_script.py")], cwd=self.base_dir
        )

    def post_process(self):
        subprocess.call(
            ["python", os.path.join(self.base_dir, "post_process.py")],
        )


def process_action(action: Actions, action_handler: ActionHandler):
    if action == Actions.get_data:
        print("Getting data...")
        action_handler.get_data()
    elif action == Actions.train:
        print("Training models...")
        action_handler.train()
    elif action == Actions.post_process:
        print("Post processing...")
        action_handler.post_process()
    else:
        raise ValueError(f"Action {action} not recognized.")
    print("Done.")


def main(args):
    if args.experiment == "predprey":
        process_action(Actions(args.action), PredPreyActionHandler())
    elif args.experiment == "lorenz":
        process_action(Actions(args.action), LorenzActionHandler())
    elif args.experiment == "mocap":
        process_action(Actions(args.action), MocapActionHandler())
    elif args.experiment == "nsde-video":
        process_action(Actions(args.action), NSDEVideoActionHandler())
    elif args.experiment == "grad-variance":
        process_action(Actions(args.action), GradVarianceActionHandler())
    else:
        raise ValueError(f"Experiment {args.experiment} not recognized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command line utility for ARCTA experiments."
    )
    exp_list = [exp.value for exp in Experiments]
    parser.add_argument(
        "experiment", type=str, choices=exp_list, help="Which experiment to run."
    )
    act_list = [act.value for act in Actions]
    parser.add_argument(
        "action", type=str, choices=act_list, help="What to do in experiment."
    )
    main(parser.parse_args())
