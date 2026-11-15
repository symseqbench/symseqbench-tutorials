"""The two networks: linear attention against its quadratic reference, causality, parameter counts.

Every test has a case it must reject.
"""

from __future__ import annotations

import pytest
import torch

from tutorial import config, models
from tutorial.ladder import build_generator

SHAPES = [
    # batch, time, heads, features, values
    (1, 1, 1, 4, 4),
    (2, 7, 1, 3, 5),
    (3, 40, 4, 8, 8),
    (4, 23, 2, 16, 4),
    (2, 40, 3, 5, 7),
]


def random_qkv(shape, seed: int, dtype=torch.float64):
    batch, time, heads, features, values = shape
    generator = torch.Generator().manual_seed(seed)
    q = models.feature_map(
        torch.randn(batch, time, heads, features, generator=generator, dtype=dtype)
    )
    k = models.feature_map(
        torch.randn(batch, time, heads, features, generator=generator, dtype=dtype)
    )
    v = torch.randn(batch, time, heads, values, generator=generator, dtype=dtype)
    return q, k, v


def largest_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).detach().abs().max())


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
def test_linear_attention_equals_the_quadratic_reference(shape):
    q, k, v = random_qkv(shape, seed=sum(shape))
    linear = models.linear_attention(q, k, v)
    assert linear.shape == v.shape
    assert largest_difference(linear, models.quadratic_attention(q, k, v, causal=True)) <= 1e-10
    # must reject: without the causal mask the reference is another function (for more than one position)
    if shape[1] > 1:
        assert largest_difference(linear, models.quadratic_attention(q, k, v, causal=False)) > 1e-3


def test_the_first_position_attends_only_to_itself():
    """Hand value: at t = 0 the weighted mean has one term, so the output is the value itself."""
    q, k, v = random_qkv((2, 9, 2, 4, 6), seed=5)
    assert largest_difference(models.linear_attention(q, k, v)[:, 0], v[:, 0]) <= 1e-12
    # must reject: the acausal reference mixes later values into the first position
    assert (
        largest_difference(models.quadratic_attention(q, k, v, causal=False)[:, 0], v[:, 0]) > 1e-3
    )


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_the_attention_module_equals_the_reference_with_the_same_weights(dtype, tolerance):
    torch.manual_seed(3)
    layer = models.CausalLinearAttention(width=32, heads=4).to(dtype)
    x = torch.randn(3, 40, 32, dtype=dtype)
    linear = layer(x)
    layer.attend = models.quadratic_attention  # same weights, all pairs
    assert largest_difference(linear, layer(x)) <= tolerance
    # must reject: the same module with the mask removed
    layer.attend = lambda q, k, v: models.quadratic_attention(q, k, v, causal=False)
    assert largest_difference(linear, layer(x)) > 1e-3


def acausal_transformer(n_symbols: int) -> torch.nn.Module:
    model = models.build_model("linear transformer", n_symbols)
    for block in model.blocks:
        block.attention.attend = lambda q, k, v: models.quadratic_attention(q, k, v, causal=False)
    return model


def change_before(model: torch.nn.Module, n_symbols: int, position: int) -> float:
    """Largest change of the scores before ``position`` when the input changes from ``position`` on."""
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(n_symbols, (4, 40), generator=generator)
    other = tokens.clone()
    other[:, position:] = (tokens[:, position:] + 1) % n_symbols
    one_hot = torch.nn.functional.one_hot
    model.eval()
    with torch.no_grad():
        a = model(one_hot(tokens, n_symbols).float())
        b = model(one_hot(other, n_symbols).float())
    assert largest_difference(a[:, position:], b[:, position:]) > 1e-3  # the change is seen later
    return largest_difference(a[:, :position], b[:, :position])


@pytest.mark.parametrize("name", config.MODELS)
def test_both_networks_are_causal(name):
    torch.manual_seed(1)
    model = models.build_model(name, n_symbols=7)
    for position in (1, 13, 39):
        assert change_before(model, 7, position) <= 1e-6
    # must reject: a transformer whose attention also looks ahead
    torch.manual_seed(1)
    assert change_before(acausal_transformer(7), 7, 13) > 1e-3


def test_both_networks_have_about_twenty_thousand_parameters_on_every_grammar():
    budget = config.PRODUCTION.parameter_budget
    assert budget == 20_000
    for rung in config.CANDIDATES.values():
        n_symbols = len(build_generator(rung).alphabet) + 1
        rnn = models.build_model("rnn", n_symbols, budget)
        transformer = models.build_model("linear transformer", n_symbols, budget)
        counts = (models.n_parameters(rnn), models.n_parameters(transformer))
        assert all(19_800 <= count <= 20_200 for count in counts), (rung.name, counts)
        assert abs(counts[0] - counts[1]) / min(counts) < 0.015, (rung.name, counts)
        # the closed formulas count the same parameters
        assert counts[0] == models.rnn_parameters(n_symbols, rnn.rnn.hidden_size)
        hidden = transformer.blocks[0].mlp[0].out_features
        assert counts[1] == models.transformer_parameters(n_symbols, 32, 2, hidden, 40)
        # the adjusted size is the closest one: one unit more or less is further from the budget
        for step in (-1, 1):
            further = models.rnn_parameters(n_symbols, rnn.rnn.hidden_size + step)
            assert abs(further - budget) >= abs(counts[0] - budget)
    # must reject: a network of another size is outside the interval
    assert not 19_800 <= models.n_parameters(models.TanhRNN(5, 100)) <= 20_200
    with pytest.raises(KeyError):
        models.build_model("lstm", 5)  # must reject: an unknown model


def test_sizes_recorded_for_the_foundation_measurement():
    """The RNN of the training-time measurement: 136 hidden units and 20 133 parameters for 5 classes."""
    rnn = models.build_model("rnn", 5)
    assert (rnn.rnn.hidden_size, models.n_parameters(rnn)) == (136, 20_133)
    assert models.free_size(rnn) == {"hidden units": 136}
    with pytest.raises(TypeError):
        models.free_size(torch.nn.Linear(2, 2))  # must reject: not a model of the tutorial


def test_scores_have_one_row_per_position_and_one_column_per_symbol():
    for name in config.MODELS:
        model = models.build_model(name, n_symbols=10)
        assert model(torch.zeros(3, 40, 10)).shape == (3, 40, 10)
        assert model(torch.zeros(2, 12, 10)).shape == (2, 12, 10)  # shorter sequences are accepted
    with pytest.raises(ValueError):
        models.CausalLinearAttention(width=30, heads=4)  # must reject: width not divisible by heads
    with pytest.raises(RuntimeError):
        models.build_model("linear transformer", 10)(
            torch.zeros(1, 41, 10)
        )  # must reject: longer than the positions it has
