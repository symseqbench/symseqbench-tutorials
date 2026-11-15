"""The two networks of the tutorial: a tanh RNN and a causal linear transformer.

Both read a sequence of one-hot symbols, shape (batch, time, symbols), and return one row of scores
(logits) per position, same shape: the scores at position t are the prediction of symbol t + 1.

## Theory

Causal attention replaces the value at position t by a weighted mean of the values up to t,

    Eq. (1)  out_t = sum_{s <= t} sim(q_t, k_s) v_s / sum_{s <= t} sim(q_t, k_s).

Softmax attention uses sim(q, k) = exp(q . k) and has to visit every pair (t, s): the cost grows
with the square of the sequence length. With a feature map phi and the kernel

    Eq. (2)  sim(q, k) = phi(q) . phi(k),          phi(x) = elu(x) + 1 > 0,

the sums over s no longer depend on t except through their upper limit, so they are running sums:

    Eq. (3)  out_t = phi(q_t)^T S_t / phi(q_t)^T z_t,
             S_t = S_{t-1} + phi(k_t) v_t^T,       z_t = z_{t-1} + phi(k_t).

The cost grows linearly with the sequence length, and the pair (S_t, z_t) is a state carried from
one position to the next, as in a recurrent network.

Ref: Katharopoulos, Vyas, Pappas and Fleuret (2020), "Transformers are RNNs: Fast Autoregressive
Transformers with Linear Attention", ICML 2020, arXiv:2006.16236. The feature map elu(x) + 1 and
the running sums are theirs.

``linear_attention`` is Eq. (3). ``quadratic_attention`` is Eq. (1) with the kernel of Eq. (2),
written over all pairs; it is used only in the tests, where the two must agree.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

# --- tanh RNN -------------------------------------------------------------------------------------


class TanhRNN(nn.Module):
    """h_t = tanh(W x_t + U h_{t-1} + b), scores_t = R h_t + c (``torch.nn.RNN``)."""

    def __init__(self, n_symbols: int, hidden: int):
        super().__init__()
        self.rnn = nn.RNN(n_symbols, hidden, nonlinearity="tanh", batch_first=True)
        self.readout = nn.Linear(hidden, n_symbols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.rnn(x)
        return self.readout(states)


# --- causal linear transformer --------------------------------------------------------------------


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """phi(x) = elu(x) + 1, positive everywhere.  Eq. (2)"""
    return F.elu(x) + 1.0


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal attention in one pass over time.  Eq. (3)

    ``q`` and ``k`` are phi(queries) and phi(keys), shape (batch, time, heads, features); ``v`` are the
    values, shape (batch, time, heads, values).
    """
    s = torch.einsum("bthf,bthd->bthfd", k, v).cumsum(dim=1)  # S_t = sum_{s <= t} phi(k_s) v_s^T
    z = k.cumsum(dim=1)  # z_t = sum_{s <= t} phi(k_s)
    numerator = torch.einsum("bthf,bthfd->bthd", q, s)  # phi(q_t)^T S_t
    denominator = torch.einsum("bthf,bthf->bth", q, z)  # phi(q_t)^T z_t
    return numerator / denominator.unsqueeze(-1)


class CausalLinearAttention(nn.Module):
    """Several attention heads side by side, each with ``width / heads`` features."""

    attend: Callable = staticmethod(linear_attention)

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError(f"width ({width}) must be a multiple of heads ({heads}).")
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        shape = (batch, time, self.heads, width // self.heads)
        q = feature_map(self.query(x).view(shape))
        k = feature_map(self.key(x).view(shape))
        v = self.value(x).view(shape)
        return self.out(self.attend(q, k, v).reshape(batch, time, width))


class Block(nn.Module):
    """Attention, then a two-layer network. Each reads a normalised copy of x and adds its output to x."""

    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = CausalLinearAttention(width, heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class LinearTransformer(nn.Module):
    """Embedding of the symbol plus an embedding of its position, ``layers`` blocks, a linear readout."""

    def __init__(
        self, n_symbols: int, width: int, heads: int, layers: int, hidden: int, max_length: int
    ):
        super().__init__()
        self.embed = nn.Linear(n_symbols, width)
        self.position = nn.Parameter(0.02 * torch.randn(max_length, width))
        self.blocks = nn.Sequential(*(Block(width, heads, hidden) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, n_symbols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x) + self.position[: x.shape[1]]
        return self.readout(self.norm(self.blocks(x)))


# --- reference for the tests ----------------------------------------------------------------------


def quadratic_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """Eq. (1) with the kernel of Eq. (2), over all pairs (t, s). Same arguments as ``linear_attention``.

    With ``causal=False`` every position also attends to the positions after it.
    """
    weights = torch.einsum("bthf,bshf->bhts", q, k)  # phi(q_t) . phi(k_s) for every pair
    if causal:
        weights = weights.tril()  # keep s <= t
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return torch.einsum("bhts,bshd->bthd", weights, v)


# --- sizes ----------------------------------------------------------------------------------------


def n_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def rnn_parameters(n_symbols: int, hidden: int) -> int:
    """Input weights, recurrent weights, two bias vectors, readout weights and bias."""
    return hidden * n_symbols + hidden * hidden + 2 * hidden + n_symbols * hidden + n_symbols


def transformer_parameters(
    n_symbols: int, width: int, layers: int, hidden: int, max_length: int
) -> int:
    """Embedding, positions, ``layers`` blocks (two norms, four projections, two-layer network), norm, readout."""
    block = 2 * 2 * width + 4 * (width * width + width) + (2 * width * hidden + hidden + width)
    embedding = width * n_symbols + width + max_length * width
    readout = 2 * width + n_symbols * width + n_symbols
    return embedding + layers * block + readout


def closest_size(count: Callable[[int], int], budget: int, largest: int = 4096) -> int:
    """The size, from 1 to ``largest``, whose parameter count is closest to ``budget``."""
    return min(range(1, largest + 1), key=lambda size: (abs(count(size) - budget), size))


def build_model(
    name: str,
    n_symbols: int,
    budget: int = 20_000,
    width: int = 32,
    heads: int = 4,
    layers: int = 2,
    max_length: int = 40,
) -> nn.Module:
    """One of the two networks, sized to ``budget`` trainable parameters.

    The number of input and output units is the number of symbols of the grammar, so one free size is
    adjusted per model: the number of hidden units of the RNN, and the number of hidden units of the
    two-layer networks inside the transformer blocks. Width, heads and layers of the transformer are
    fixed.
    """
    if name == "rnn":
        hidden = closest_size(lambda size: rnn_parameters(n_symbols, size), budget)
        return TanhRNN(n_symbols, hidden)
    if name == "linear transformer":
        hidden = closest_size(
            lambda size: transformer_parameters(n_symbols, width, layers, size, max_length), budget
        )
        return LinearTransformer(n_symbols, width, heads, layers, hidden, max_length)
    raise KeyError(f"Unknown model {name!r}. Known models: 'rnn', 'linear transformer'.")


def free_size(model: nn.Module) -> dict[str, int]:
    """The size that ``build_model`` adjusted, for the record of a run."""
    if isinstance(model, TanhRNN):
        return {"hidden units": model.rnn.hidden_size}
    if isinstance(model, LinearTransformer):
        return {"hidden units per block": model.blocks[0].mlp[0].out_features}
    raise TypeError(f"Not a model of the tutorial: {type(model).__name__}.")


def size_table(n_symbols: dict[str, int], **sizes) -> list[dict]:
    """Per grammar: number of classes, and the adjusted size and parameter count of each model."""
    rows = []
    for grammar, classes in n_symbols.items():
        rnn = build_model("rnn", classes, **sizes)
        transformer = build_model("linear transformer", classes, **sizes)
        rows.append(
            {
                "grammar": grammar,
                "classes": classes,
                "RNN: hidden units": rnn.rnn.hidden_size,
                "RNN: parameters": n_parameters(rnn),
                "transformer: hidden units per block": transformer.blocks[0].mlp[0].out_features,
                "transformer: parameters": n_parameters(transformer),
            }
        )
    return rows
