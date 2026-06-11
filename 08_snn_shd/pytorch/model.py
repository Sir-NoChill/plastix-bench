"""SNN model variants and a parameter-matched RNN baseline.

Three architectures, picked via `--model` in train.py:

- `rsnn`: Linear(700 -> H) -> RLeaky(H, recurrent) -> Linear(H -> 20) -> Leaky.
  Membrane potential of the readout LIF is accumulated across time and used
  as the logits (matches snnTorch's `ce_rate_loss` / `ce_count_loss`
  recipes; we use cross-entropy on summed membrane).
- `fsnn`: same shape but the hidden layer is a plain `Leaky` (no recurrence)
  — quantifies how much recurrence buys.
- `gru`: nn.GRU(700 -> H) -> Linear(H -> 20) on the *same binned input*.
  Hidden size is solved so the parameter count matches the SNN; this is
  the apples-to-apples non-spiking baseline (spec section 6.1).

All three accept the same (T, B, 700) input tensor and return (B, 20)
logits, so the training/eval loops don't need to know which one they hold.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _spike_grad(slope: float = 25.0):
    from snntorch import surrogate
    return surrogate.fast_sigmoid(slope=slope)


class RecurrentSNN(nn.Module):
    """Recurrent LIF network with snnTorch RLeaky.

    The readout LIF uses `reset_mechanism="none"` so its membrane is a
    clean accumulator (no spike-based reset). Cross-entropy on the
    summed readout membrane then has a non-pathological gradient even
    if hidden firing is sparse. This is the standard SHD recipe and the
    fix for the "dead network at chance accuracy" pitfall in the spec."""

    def __init__(self, n_in: int = 700, n_hid: int = 256, n_out: int = 20,
                 beta: float = 0.9, learn_beta: bool = True,
                 surrogate_slope: float = 25.0):
        super().__init__()
        import snntorch as snn
        sg = _spike_grad(surrogate_slope)
        self.n_hid = n_hid
        self.fc_in = nn.Linear(n_in, n_hid)
        # SHD inputs are sparse binary spikes (~1% active channels). The
        # default Linear init keeps pre-activations near 0.04 * n_active
        # which lands well below the LIF threshold (1.0), starving the
        # hidden layer. Kaiming-uniform with fan_out scaling gives a
        # large enough drive to actually fire on init.
        nn.init.kaiming_uniform_(self.fc_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_in.bias)
        self.rlif = snn.RLeaky(beta=beta, linear_features=n_hid,
                               spike_grad=sg, learn_beta=learn_beta)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg,
                                 learn_beta=learn_beta,
                                 reset_mechanism="none")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, n_in). Returns (logits, mean_firing_rate)."""
        T = x.shape[0]
        spk_h, mem_h = self.rlif.init_rleaky()
        mem_o = self.lif_out.init_leaky()
        # Hidden spike count for rate regularization + reporting.
        spike_sum = torch.zeros((), device=x.device)
        elems = 0
        logits = 0.0
        for t in range(T):
            cur_h = self.fc_in(x[t])
            spk_h, mem_h = self.rlif(cur_h, spk_h, mem_h)
            cur_o = self.fc_out(spk_h)
            spk_o, mem_o = self.lif_out(cur_o, mem_o)
            logits = logits + mem_o
            spike_sum = spike_sum + spk_h.sum()
            elems += spk_h.numel()
        firing_rate = spike_sum / max(elems, 1)
        return logits, firing_rate


class FeedforwardSNN(nn.Module):
    """Same shape as RecurrentSNN but the hidden layer is non-recurrent."""

    def __init__(self, n_in: int = 700, n_hid: int = 256, n_out: int = 20,
                 beta: float = 0.9, learn_beta: bool = True,
                 surrogate_slope: float = 25.0):
        super().__init__()
        import snntorch as snn
        sg = _spike_grad(surrogate_slope)
        self.n_hid = n_hid
        self.fc_in = nn.Linear(n_in, n_hid)
        nn.init.kaiming_uniform_(self.fc_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_in.bias)
        self.lif_h = snn.Leaky(beta=beta, spike_grad=sg, learn_beta=learn_beta)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg,
                                 learn_beta=learn_beta,
                                 reset_mechanism="none")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[0]
        mem_h = self.lif_h.init_leaky()
        mem_o = self.lif_out.init_leaky()
        spike_sum = torch.zeros((), device=x.device)
        elems = 0
        logits = 0.0
        for t in range(T):
            cur_h = self.fc_in(x[t])
            spk_h, mem_h = self.lif_h(cur_h, mem_h)
            cur_o = self.fc_out(spk_h)
            spk_o, mem_o = self.lif_out(cur_o, mem_o)
            logits = logits + mem_o
            spike_sum = spike_sum + spk_h.sum()
            elems += spk_h.numel()
        firing_rate = spike_sum / max(elems, 1)
        return logits, firing_rate


class GRUBaseline(nn.Module):
    """Single-layer GRU + linear readout. Returns (logits, 0.0) so the
    training loop can stay agnostic between SNN and non-SNN outputs."""

    def __init__(self, n_in: int = 700, n_hid: int = 256, n_out: int = 20):
        super().__init__()
        self.gru = nn.GRU(input_size=n_in, hidden_size=n_hid,
                          batch_first=False)
        self.fc_out = nn.Linear(n_hid, n_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, n_in). GRU output: (T, B, H).
        out, _ = self.gru(x)
        # Match SNN's "sum over time" readout so the comparison stays clean.
        logits = self.fc_out(out.sum(dim=0))
        return logits, torch.zeros((), device=x.device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_gru_hidden(n_in: int, n_out: int, target_params: int) -> int:
    """Solve for the GRU hidden size that gets closest to `target_params`.

    Parameter count of nn.GRU(input, H, num_layers=1, bias=True):
        3 * (input*H + H*H + 2*H) = 3*(input + H + 2) * H
    Plus the readout Linear(H, out, bias=True):
        H * out + out
    Total: 3*(input + H + 2)*H + H*out + out

    Solve quadratic: 3*H^2 + (3*input + 6 + out)*H + (out - target) = 0.
    """
    a = 3.0
    b = 3.0 * n_in + 6.0 + n_out
    c = float(n_out - target_params)
    disc = b * b - 4 * a * c
    if disc < 0:
        return 1
    root = (-b + math.sqrt(disc)) / (2 * a)
    return max(1, int(round(root)))


def build_model(name: str, n_in: int, n_hid: int, n_out: int, beta: float,
                surrogate_slope: float, match_to: nn.Module | None = None):
    """Factory. If `match_to` is set and `name == 'gru'`, the GRU's hidden
    size is recomputed to match the target's parameter count."""
    if name == "rsnn":
        return RecurrentSNN(n_in, n_hid, n_out, beta=beta,
                            surrogate_slope=surrogate_slope)
    if name == "fsnn":
        return FeedforwardSNN(n_in, n_hid, n_out, beta=beta,
                              surrogate_slope=surrogate_slope)
    if name == "gru":
        if match_to is not None:
            n_hid = match_gru_hidden(n_in, n_out, count_params(match_to))
        return GRUBaseline(n_in, n_hid, n_out)
    raise ValueError(f"unknown model: {name}")
