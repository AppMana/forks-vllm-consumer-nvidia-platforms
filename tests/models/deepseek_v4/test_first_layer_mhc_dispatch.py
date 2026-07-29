# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The first decoder layer must not read the token count as ``hc_mult``.

V4 carries ``hc_mult`` parallel residual streams, so the decoder layers work on
``(T, hc_mult, H)`` while the embedding is ``(T, H)``. Every MHC kernel infers
``hc_mult`` from ``residual.shape[-2]``. Upstream #48137 replaced the
unconditional expand with a rank dispatch (a 2-D first-layer branch that either
runs the tilelang broadcast kernel or materializes the streams); the merge kept
only the 3-D side, so the 2-D embedding fell into the path where
``hc_mult = residual.shape[-2]`` is the number of tokens (02802b3037).

``T`` and ``hc_mult`` are deliberately different and coprime here, so a residual
carrying the wrong one cannot pass by coincidence.

No GPU: the MHC ops, attention and FFN are replaced with recorders, and only
the dispatch and the shapes it produces are under test.
"""

import ast
import types
from pathlib import Path

import pytest
import torch

NUM_TOKENS = 7
HC_MULT = 4
HIDDEN_SIZE = 16

REPO_ROOT = Path(__file__).resolve().parents[3]


class _Recorder:
    """Records the residual each MHC entry point is handed."""

    def __init__(self):
        self.mhc_pre_residuals: list[torch.Size] = []
        self.broadcast_inputs: list[torch.Size] = []
        self.attn_norm_calls = 0


def _make_layer(recorder: _Recorder, *, hc_attn_fn_broadcast=None):
    """A DeepseekV4DecoderLayer receiver with only the MHC wiring real."""
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer

    layer = types.SimpleNamespace()
    layer.hc_mult = HC_MULT
    layer.rms_norm_eps = 1e-6
    layer.hc_eps = 1e-6
    layer.hc_post_alpha = 2.0
    layer.hc_sinkhorn_iters = 3
    layer.hc_attn_fn_broadcast = hc_attn_fn_broadcast
    for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base"):
        setattr(layer, name, torch.zeros(1))
    for name in ("hc_attn_scale", "hc_ffn_scale"):
        setattr(layer, name, torch.zeros(3))

    def mhc_pre(*, residual, **kwargs):
        recorder.mhc_pre_residuals.append(residual.shape)
        num_tokens = residual.shape[0]
        return (
            torch.zeros(num_tokens, HC_MULT),
            torch.zeros(num_tokens, HC_MULT, HC_MULT),
            residual,
        )

    def attn_norm(x):
        recorder.attn_norm_calls += 1
        return x

    attn_norm.weight = torch.ones(HIDDEN_SIZE)
    attn_norm.variance_epsilon = 1e-6

    def mhc_fused_post_pre(x, residual, post_mix, res_mix, *args):
        return residual, post_mix, res_mix, x

    layer.mhc_pre = mhc_pre
    layer.hc_pre = types.MethodType(DeepseekV4DecoderLayer.hc_pre, layer)
    layer.attn_norm = attn_norm
    layer.attn = lambda positions, x, _: x
    layer.mhc_fused_post_pre = mhc_fused_post_pre
    layer.ffn_norm = lambda x: x
    layer.ffn = lambda x, input_ids: x
    return layer


def _forward(layer, x):
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer

    return DeepseekV4DecoderLayer.forward(
        layer, x, positions=torch.arange(NUM_TOKENS), input_ids=None
    )


def test_two_dimensional_first_layer_input_never_infers_hc_mult_from_tokens(
    monkeypatch,
) -> None:
    """The fallback path materializes ``(T, hc_mult, H)`` before mhc_pre.

    Would have caught 02802b3037: without the ``x.dim() == 2`` branch the
    embedding reached mhc_pre as ``(T, H)``, where ``residual.shape[-2]``
    is ``T``.
    """
    from vllm.models.deepseek_v4.nvidia import model as model_module

    # sm_8x has no broadcast kernel and takes the materializing branch.
    monkeypatch.setattr(model_module, "mhc_uses_tilelang", lambda: False)

    recorder = _Recorder()
    layer = _make_layer(recorder)
    _forward(layer, torch.zeros(NUM_TOKENS, HIDDEN_SIZE))

    assert recorder.mhc_pre_residuals, "mhc_pre was never reached"
    for shape in recorder.mhc_pre_residuals:
        assert len(shape) == 3, f"mhc_pre got a rank-{len(shape)} residual: {shape}"
        assert shape[-2] == HC_MULT, (
            f"mhc_pre residual is {tuple(shape)}; shape[-2] is the value every "
            f"MHC kernel reads as hc_mult, and it must be {HC_MULT}, not the "
            f"token count {NUM_TOKENS}"
        )
        assert shape[0] == NUM_TOKENS
        assert shape[-1] == HIDDEN_SIZE

    # The materializing path does not fold the norm, so it is applied once.
    assert recorder.attn_norm_calls == 1


def test_two_dimensional_first_layer_input_takes_the_broadcast_kernel(
    monkeypatch,
) -> None:
    """With tilelang and the broadcast weight present, use the fused kernel.

    ``hc_attn_fn_broadcast`` was computed and never read after the merge --
    the tell that the dispatch had been dropped. The broadcast kernel folds
    RMSNorm, so the unfused ``attn_norm`` must be suppressed on this path only.
    """
    from vllm.model_executor.kernels.mhc import tilelang as tilelang_module
    from vllm.models.deepseek_v4.nvidia import model as model_module

    recorder = _Recorder()

    def fake_broadcast(x, *args, **kwargs):
        recorder.broadcast_inputs.append(x.shape)
        residual = x.unsqueeze(-2).expand(-1, HC_MULT, -1).contiguous()
        return (
            residual,
            torch.zeros(NUM_TOKENS, HC_MULT),
            torch.zeros(NUM_TOKENS, HC_MULT, HC_MULT),
            residual,
        )

    monkeypatch.setattr(model_module, "mhc_uses_tilelang", lambda: True)
    monkeypatch.setattr(
        tilelang_module, "mhc_pre_broadcast_tilelang", fake_broadcast, raising=False
    )

    layer = _make_layer(recorder, hc_attn_fn_broadcast=torch.zeros(1))
    _forward(layer, torch.zeros(NUM_TOKENS, HIDDEN_SIZE))

    assert recorder.broadcast_inputs == [torch.Size([NUM_TOKENS, HIDDEN_SIZE])]
    assert not recorder.mhc_pre_residuals, (
        "the broadcast path must not also run the unfused mhc_pre"
    )
    assert recorder.attn_norm_calls == 0, (
        "the broadcast kernel folds RMSNorm; applying attn_norm again normalizes twice"
    )


def test_later_layers_keep_the_fused_post_pre_path(monkeypatch) -> None:
    """A layer handed a residual must not re-enter the first-layer branch."""
    from vllm.models.deepseek_v4.nvidia import model as model_module

    monkeypatch.setattr(model_module, "mhc_uses_tilelang", lambda: False)

    recorder = _Recorder()
    layer = _make_layer(recorder)
    residual = torch.zeros(NUM_TOKENS, HC_MULT, HIDDEN_SIZE)
    _forward_kwargs = dict(
        post_mix=torch.zeros(NUM_TOKENS, HC_MULT),
        res_mix=torch.zeros(NUM_TOKENS, HC_MULT, HC_MULT),
        residual=residual,
    )
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer

    DeepseekV4DecoderLayer.forward(
        layer,
        torch.zeros(NUM_TOKENS, HC_MULT, HIDDEN_SIZE),
        positions=torch.arange(NUM_TOKENS),
        input_ids=None,
        **_forward_kwargs,
    )

    assert not recorder.mhc_pre_residuals
    assert recorder.attn_norm_calls == 1


# ---------------------------------------------------------------------------
# mHC parameter shapes
# ---------------------------------------------------------------------------


def _assigned_expression(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    expressions = {
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    }
    assert len(expressions) == 1, f"{path.name}: {name} = {expressions}"
    return expressions.pop()


@pytest.mark.parametrize("backend", ["nvidia", "amd", "xpu"])
def test_mix_hc_matches_what_the_mhc_kernels_expect(backend: str) -> None:
    """``fn.shape[0]`` must be the ``hc_mult3`` the kernel derives.

    ``mhc_pre_tilelang`` infers ``hc_mult`` from the residual and then asserts
    ``fn.shape[0] == hc_mult * 2 + hc_mult * hc_mult``. The three model
    backends each allocate ``hc_attn_fn`` with their own ``mix_hc``
    expression, so this pins all four to one number -- and it is the same
    assert a 2-D first-layer residual trips, since a token count for
    ``hc_mult`` makes the expected shape enormous.
    """
    model_py = REPO_ROOT / f"vllm/models/deepseek_v4/{backend}/model.py"
    if not model_py.exists():
        pytest.skip(f"no {backend} backend in this tree")

    mix_hc_expr = _assigned_expression(model_py, "mix_hc")
    kernel_expr = _assigned_expression(
        REPO_ROOT / "vllm/model_executor/kernels/mhc/tilelang.py", "hc_mult3"
    )

    for hc_mult in range(1, 9):
        scope = {"self": types.SimpleNamespace(hc_mult=hc_mult), "hc_mult": hc_mult}
        scope["hc_mult2"] = hc_mult * hc_mult
        mix_hc = eval(mix_hc_expr, {}, scope)  # noqa: S307 - repo source only
        expected = eval(kernel_expr, {}, scope)  # noqa: S307
        assert mix_hc == expected == (2 + hc_mult) * hc_mult, (
            f"{backend}: mix_hc={mix_hc_expr} gives {mix_hc} at hc_mult="
            f"{hc_mult}, but the MHC kernels assert fn.shape[0] == {expected}"
        )
