# Amortized reparametrization: Efficient and Scalable Variational Inference for Latent SDEs

Accompanying code for the [NeurIPS 2023
paper](https://openreview.net/forum?id=5yZiP9fZNv)
by Kevin Course and Prasanth B. Nair.

**Tutorials and documentation coming soon!**

## 1. Installation

### Installing the package

The package can be installed through PyPI:

```bash
pip install arlatentsde
```

### Reproducing the experiment environment

We ran experiments on a Linux machine with CUDA 11.8.
We used [poetry](https://github.com/python-poetry/poetry) to manage dependencies.

If you prefer a different environment manager, all dependencies are listed
in the `pyproject.toml`.

To reproduce the experiment environment, first navigate to branch named
`neurips-freeze`.
Then install all optional dependencies required to run experiments,

```bash
poetry install --with dev,exps
```

To download all pretrained models, datasets, and figures we use [repopacker](https://github.com/coursekevin/repopacker):

```bash
repopacker download models-data-figs.zip
repopacker unpack models-data-figs.zip
```

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

## 3. Reference

Course, K., Nair, P.B. Amortized Reparametrization: Efficient and Scalable Variational Inference for Latent SDEs.  
In Proc. Advances in Neural Information Processing Systems, (2023).
