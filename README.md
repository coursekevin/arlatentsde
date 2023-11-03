# Amortized reparametrization: Efficient and Scalable Variational Inference for Latent SDEs

## 1. Installation

We ran experiments on a Linux machine with CUDA 11.8.
We used [poetry](https://github.com/python-poetry/poetry) to manage dependencies.

If you prefer a different environment manager, all dependencies are listed
in the `pyproject.toml`. Otherwise, you can reproduce our environment as follows:

1. If you don't have an environment with CUDA 11.8 installed, you can initialize a
   conda environment with necessary dependencies using [conda](https://docs.conda.io/en/latest/):

```bash
conda env create -f base-env.yml
```

2. To install all necessary dependencies required to run experiments run,

```bash
poetry install --with dev,exps
```

This will install dependencies according to the lock file `poetry.lock`.

## 2. Usage

The numerical studies can be rerun from the experiments
directory using the command-line script `main.py`. All numerical
studies follow the same basic structure:
(i) generate / download,
(ii) train model, and
(iii) post process for plots and tables.

The script has the following syntax:

```bash
python main.py [experiment] [action]
```

The choices of experiments and actions are provided below:

- Experiments:
  - `predprey`: Orders of magnitude magnitude fewer NFEs experiment
  - `lorenz`: Adjoint instabilities experiment
  - `mocap`: Motion capture benchmark
  - `nsde-video`: Neural SDE from video experiment
  - `grad-variance`: Gradient variance experiment
- Actions:
  - `get-data`: Download / generate data
  - `train`: Train models
  - `post-process`: Post process for plots and tables
