"""Three sequence models for the Compact C. elegans-Wired Controller (CCWC)
benchmark, spec section 4:

- A — wired NCP: LTC (or CfC) over an `AutoNCP` sparse topology.
- B — dense LTC: LTC (or CfC) over a fully-connected hidden layer with the
       same neuron count as A. Isolates the wiring contribution.
- C — LSTM baseline: a single-layer LSTM sized so the parameter count is
       5-10x larger than A.

All three return logits `(B, n_out)` for classification or signal `(B, n_out)`
/ `(B, T, n_out)` for regression. The classification head and last-step
selection live inside the model so the training loop stays uniform.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP


# ---------------------------------------------------------------------------
# Wrappers around ncps models — pick variant + add a head when wiring's
# motor count doesn't already match the task output dim.
# ---------------------------------------------------------------------------

def _build_ncp_core(variant: str, input_size: int, units, *,
                    return_sequences: bool, mixed_memory: bool):
    """Instantiate either an LTC or a CfC with the given wiring spec.

    `units` is either an `AutoNCP` wiring (A) or an int (B). The function
    is just a thin switch so the rest of the module stays variant-agnostic.
    """
    if variant == "ltc":
        return LTC(input_size=input_size, units=units,
                   batch_first=True, return_sequences=return_sequences,
                   mixed_memory=mixed_memory)
    if variant == "cfc":
        return CfC(input_size=input_size, units=units,
                   batch_first=True, return_sequences=return_sequences,
                   mixed_memory=mixed_memory)
    raise ValueError(f"unknown ncp variant: {variant!r}")


class WiredNCPModel(nn.Module):
    """Model A. Sparse AutoNCP wiring + LTC/CfC dynamics.

    The wiring's motor neuron count is set to `n_out`, so the core already
    emits the right shape — no extra head needed. We still keep a thin
    identity layer in the spot a head would go, to keep the forward path
    uniform with the other two models.
    """

    def __init__(self, input_size: int, units: int, n_out: int,
                 task: str, variant: str = "ltc",
                 mixed_memory: bool = False):
        super().__init__()
        self.task = task
        self.n_hid = units
        wiring = AutoNCP(units, n_out)
        self.core = _build_ncp_core(
            variant, input_size, wiring,
            return_sequences=(task == "sine"),
            mixed_memory=mixed_memory,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.core(x)
        return y


class DenseLTCModel(nn.Module):
    """Model B. Fully-connected LTC/CfC with the same neuron count as A.

    `units` is passed as an int to ncps, which means a dense interconnect.
    Output is the full hidden state (`B, T, units` or `B, units`); a small
    linear head maps to `n_out`.
    """

    def __init__(self, input_size: int, units: int, n_out: int,
                 task: str, variant: str = "ltc",
                 mixed_memory: bool = False):
        super().__init__()
        self.task = task
        self.n_hid = units
        self.core = _build_ncp_core(
            variant, input_size, units,
            return_sequences=(task == "sine"),
            mixed_memory=mixed_memory,
        )
        self.head = nn.Linear(units, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.core(x)
        # y is (B, T, H) for sine, (B, H) for psmnist; Linear handles both.
        return self.head(y)


class LSTMBaselineModel(nn.Module):
    """Model C. Single-layer LSTM + linear head. Sized to ~5-10x A's params.

    For classification (psmnist) we read off the last-step hidden state;
    for regression (sine) we return the full sequence — matching the LTC
    output shapes above so the training loop stays uniform.
    """

    def __init__(self, input_size: int, hidden_size: int, n_out: int,
                 task: str):
        super().__init__()
        self.task = task
        self.n_hid = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.head = nn.Linear(hidden_size, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        if self.task == "sine":
            return self.head(out)          # (B, T, n_out)
        return self.head(out[:, -1, :])    # (B, n_out)


# ---------------------------------------------------------------------------
# Factory + helpers
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, *, input_size: int, units: int, n_out: int,
                task: str, variant: str = "ltc",
                lstm_hidden: int = 128,
                mixed_memory: bool = False) -> nn.Module:
    """Factory used by train.py and robustness.py.

    `name` is one of {"A","B","C"}. `units` is the LTC/CfC neuron count
    (shared between A and B); `lstm_hidden` sizes the LSTM baseline.
    `variant` switches LTC <-> CfC for A and B; spec section 9 recommends
    swapping to CfC if LTC is too slow.
    """
    if name == "A":
        return WiredNCPModel(input_size, units, n_out, task,
                             variant=variant, mixed_memory=mixed_memory)
    if name == "B":
        return DenseLTCModel(input_size, units, n_out, task,
                             variant=variant, mixed_memory=mixed_memory)
    if name == "C":
        return LSTMBaselineModel(input_size, lstm_hidden, n_out, task)
    raise ValueError(f"unknown model: {name!r} (expected A, B, or C)")
