import os
from copy import deepcopy
from multiprocessing import get_context
from functools import partial
from typing import Callable, List, cast, Protocol

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import default_collate
from jaxtyping import Float
from pytorch_lightning.callbacks import Callback
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from ..autoencoders import Decoder, Encoder
from ..latent_sdes import LatentSDE
from ..data import EvenlySpacedFrames, EvenlySpacedTensors
from ..elbo import rsample_latent
from ..integrate import solve_sde
from ..priors import Prior


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.cpu() for k, v in model.state_dict().items()}


class SummaryCallback(Protocol):
    def __call__(
        self,
        fig_dir: str,
        t: Float[Tensor, " n"],
        xwin: Float[Tensor, "n c ..."],
        x: Float[Tensor, "n ..."],
        iter_count: int,
        encoder: Encoder,
        decoder: Decoder,
        drift: LatentSDE,
        brownian: Prior,
    ) -> None:
        ...


class ProgressSummarizer:
    def __init__(
        self,
        nlatents: int,
        fig_dir: str,
        dset: EvenlySpacedFrames | EvenlySpacedTensors,
        encoder: Encoder,
        decoder: Decoder,
        drift: LatentSDE,
        brownian: Prior,
        summary_callbacks: List[SummaryCallback],
        max_process: int = 4,
    ) -> None:
        # confirm that encoder, decoder, drift, and brownian are all nn.Modules
        self._check_args(encoder, decoder, drift, brownian)
        # use cast to make the type checker happy, there is no effect at runtime.
        self.encoder = cast(nn.Module, deepcopy(encoder))
        self.decoder = cast(nn.Module, deepcopy(decoder))
        self.drift = cast(nn.Module, deepcopy(drift))
        self.brownian = cast(nn.Module, deepcopy(brownian))
        idx = (torch.linspace(0, 1, nlatents) * len(dset)).to(torch.int)  # type: ignore
        self.t, self.xwin, self.x = default_collate([dset[i] for i in idx])
        self.cbs = []
        for cb in summary_callbacks:
            self.cbs.append(partial(cb, fig_dir, self.t, self.xwin, self.x))
        self._ctx = get_context("spawn")
        self._process_list = []
        self._max_processes = max_process

    def _check_args(self, *args):
        for arg in args:
            if not isinstance(arg, nn.Module):
                raise TypeError(
                    f"{arg} must be a torch.nn.Module to use ProgressSummarizer"
                )

    def bg_plot_summary(self, iter_count, encoder, decoder, drift, brownian):
        self.wait_for_processes()
        sum_process = self._ctx.Process(
            target=self.plot_latent_from_sdicts,
            args=(
                cpu_state_dict(encoder),
                cpu_state_dict(decoder),
                cpu_state_dict(drift),
                cpu_state_dict(brownian),
                iter_count,
            ),
        )
        sum_process.daemon = True  # kill process if parent dies
        sum_process.start()
        self._process_list.append(sum_process)
        return sum_process

    def print_summary(self, iter_count: int, epoch: int, loss: float) -> None:
        print(f"epoch: {epoch:03d} | iter: {iter_count:04d} | loss: {loss:.2e}")

    def __getstate__(self):
        d = dict(self.__dict__)
        # can't pickle the process list
        del d["_process_list"]
        return d

    def wait_for_processes(self):
        while len(self._process_list) > self._max_processes:
            self._process_list.pop(0).join()

    def wait_for_all_processes(self):
        while len(self._process_list) > 0:
            self._process_list.pop(0).join()

    def plot_latent_from_sdicts(
        self, enc_sdict, dec_sdict, drift_sdict, brownian_sdict, iter_count
    ):
        self.encoder.load_state_dict(enc_sdict)
        self.decoder.load_state_dict(dec_sdict)
        self.drift.load_state_dict(drift_sdict)
        self.brownian.load_state_dict(brownian_sdict)
        self.encoder.eval()
        self.decoder.eval()
        self.drift.eval()
        self.brownian.eval()
        for cb in self.cbs:
            cb(iter_count, self.encoder, self.decoder, self.drift, self.brownian)
        self.encoder.train()
        self.decoder.train()
        self.drift.train()
        self.brownian.train()


def plot_latent(fig_dir, t, xwin, x, iter_count, encoder, decoder, drift_fn, brownian):
    raise NotImplementedError("plot_latent is deprecated")
    fig, ax = plt.subplots(squeeze=False)
    ax = ax[0, 0]
    os.makedirs(fig_dir, exist_ok=True)
    with torch.no_grad():
        # plotting encoded latent states
        # t_node, z, _ = encoder(xwin)
        z_eval = torch.cat(
            [fn(t) for t, fn, _ in tuple(encoder(xi.unsqueeze(0)) for xi in xwin)], 0
        )
        mu, _ = z_eval.chunk(2, dim=-1)
        # mu, _ = z(t_node).chunk(2, dim=-1)
        for j in range(mu.shape[-1]):
            ax.plot(t.cpu(), mu[:, j].cpu(), "o-", alpha=0.5)

        # plotting inferred latent states
        def ode(t, z):
            z = torch.as_tensor(z, dtype=torch.get_default_dtype())
            return drift_fn(z, t).cpu().numpy()

        t_eval = torch.linspace(t.min(), t.max(), 300)
        t_span = (t_eval[0], t_eval[-1])
        x0 = mu[0].cpu().to(torch.float64).numpy()
        sol = solve_ivp(ode, t_span, x0, t_eval=t_eval)
        ax.plot(sol.t, sol.y.T, "C0--", alpha=0.5)
        fig.savefig(os.path.join(fig_dir, f"latent_{iter_count:04d}.png"))  # type: ignore
    plt.close(fig)


def plot_latent_sde(
    plot_recon, fig_dir, t, xwin, x, iter_count, encoder, decoder, drift_fn, brownian
):
    fig, ax = plt.subplots(squeeze=False)
    ax = ax[0, 0]
    os.makedirs(fig_dir, exist_ok=True)
    max_plots = 10
    with torch.no_grad():
        # plotting encoded latent states
        z_eval = []
        for ti, xi in zip(t, xwin):
            _, fn, _ = encoder(ti.unsqueeze(0), xi.unsqueeze(0))
            z_eval.append(fn(ti))
        z_eval = torch.cat(z_eval, 0)
        mu, logvar = z_eval.chunk(2, dim=-1)
        n_plots = min(mu.shape[-1], max_plots)
        if plot_recon:
            fig, axs = plt.subplots(n_plots, 2)
        else:
            fig, axs = plt.subplots(n_plots)
        n_samples = 128
        z0 = rsample_latent(n_samples, mu[0], logvar[0])
        sde_kwargs = dict(dt=1e-2, adaptive=True, rtol=1e-4, atol=1e-6)
        latent_soln = solve_sde(drift_fn, brownian, z0, t, **sde_kwargs)
        if plot_recon:
            mu_full, var_full = decoder(latent_soln)
            logvar_full = torch.log(var_full)
            full_soln = mu_full + torch.randn_like(mu_full) * torch.exp(
                0.5 * logvar_full
            )
            iter_args = zip(axs.T, [latent_soln, full_soln], [mu, xwin[:, 0]])
        else:
            iter_args = zip([axs], [latent_soln], [mu])
        fill_color = "#9ebcda"
        sample_colors = ("#8c96c6", "#8c6bb1", "#810f7c")
        alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        alphas = [a + 0.1 for a in alphas]
        percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5]

        for ax, soln, data in iter_args:
            for j in range(n_plots):
                ax[j].plot(t.cpu(), data[:, j].cpu(), "k-.", alpha=0.5)

            for quant, alpha in zip(percentiles, alphas):
                lb = torch.quantile(soln, 1 - quant, dim=0)
                ub = torch.quantile(soln, quant, dim=0)
                # for lb_, ub_ in zip(lb.T, ub.T):
                for j in range(n_plots):
                    lb_, ub_ = lb.T[j], ub.T[j]
                    ax[j].fill_between(t, lb_, ub_, alpha=alpha, color=fill_color)  # type: ignore

            for k in range(n_plots):
                for j, color in enumerate(sample_colors):
                    ax[k].plot(t, soln[j, :, k], color=color, linewidth=1.0)

        fig.savefig(os.path.join(fig_dir, f"latent_{iter_count:04d}.png"), dpi=200)  # type: ignore
    plt.close(fig)


def plot_reconstruction(
    fig_dir, t, xwin, x, iter_count, encoder, decoder, drift_fn, brownian
):
    os.makedirs(fig_dir, exist_ok=True)
    # t_node, z, _ = encoder(xwin)
    # mu, _ = z(t_node).chunk(2, dim=-1)
    z_eval = []
    for ti, xi in zip(t, xwin):
        _, fn, _ = encoder(ti.unsqueeze(0), xi.unsqueeze(0))
        z_eval.append(fn(ti))
    z_eval = torch.cat(z_eval, 0)
    mu, _ = z_eval.chunk(2, dim=-1)
    logits = decoder(mu)
    num_figs = 5
    fig, ax = plt.subplots(2, num_figs)
    with torch.no_grad():
        for j, (logit, xi) in enumerate(zip(logits[:num_figs], x[:num_figs])):
            ax[0, j].imshow(xi.cpu(), cmap="gray")
            ax[1, j].imshow(torch.sigmoid(logit).cpu(), cmap="gray")
            plt.axis("off")
    fig.savefig(os.path.join(fig_dir, f"recon_{iter_count:04d}.png"))  # type: ignore
    plt.close(fig)


class ProgressSummarizerCallback(Callback):
    def __init__(self, summary_freq: int, summarizer: ProgressSummarizer):
        self.summary_freq = summary_freq
        self.summarizer = summarizer

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.summary_freq == 0:
            self.summarizer.bg_plot_summary(
                trainer.global_step,
                pl_module.encoder,
                pl_module.decoder,
                pl_module.drift,
                pl_module.brownian,
            )

    def on_fit_end(self, trainer, pl_module):
        self.summarizer.wait_for_all_processes()
        super().on_fit_end(trainer, pl_module)
