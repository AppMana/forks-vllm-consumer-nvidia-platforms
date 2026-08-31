import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "vllm" / "layer_partition.py"
SPEC = importlib.util.spec_from_file_location("layer_partition", MODULE_PATH)
assert SPEC and SPEC.loader
partition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partition)

compute_layer_counts = partition.compute_layer_counts
compute_layer_range = partition.compute_layer_range
select_shards = partition.select_shards


# The DSV4 int4/int8 DSpark rebuild (appmana/deepseek-v4-int4-int8): 43 target
# layers, 3 MTP stages worth ~4708/1571 = 3.0 layer-equivalents of weights.
DSV4_LAYERS = 43
DSV4_MTP_COST = 3.0


def test_compute_layer_counts_matches_vllm_pp_policy():
    assert compute_layer_counts(7, 3) == [2, 3, 2]
    assert compute_layer_counts(7, 4) == [2, 2, 2, 1]
    assert compute_layer_counts(43, 10) == [4, 4, 4, 4, 4, 4, 5, 5, 5, 4]
    assert compute_layer_counts(61, 12) == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5]


def test_compute_layer_counts_non_draft_unchanged_with_flag_default():
    # Non-draft behavior must stay byte-identical to
    # vllm.distributed.utils.get_pp_indices' default branch.
    assert compute_layer_counts(43, 12) == [3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3]
    assert compute_layer_counts(43, 10, draft_zero_last=False) == [
        4, 4, 4, 4, 4, 4, 5, 5, 5, 4
    ]


def test_compute_layer_counts_draft_zero_last_pp12_reference():
    # Keep one target block on the MTP/head rank so the ordinary transformer
    # pipeline and draft path do not become separate zero-layer stages.
    assert compute_layer_counts(43, 12, draft_zero_last=True) == [
        3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1
    ]


def test_mtp_partition_keeps_a_cost_neutral_transformer_seam():
    """The MTP seam must not be a zero-layer pipeline stage.

    When doing so does not raise the peak rank cost, keep the final target block
    beside MTP instead of turning the last rank into a draft-only stage. Moving
    only the final boundary makes the PP=12 experiment a controlled
    ``...,4,0`` to ``...,3,1`` comparison.
    """
    counts = compute_layer_counts(DSV4_LAYERS, 12, mtp_cost=DSV4_MTP_COST)
    assert counts == [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1]
    assert all(count >= 1 for count in counts)


def test_compute_layer_counts_draft_zero_last_pp10_properties():
    counts = compute_layer_counts(43, 10, draft_zero_last=True)
    assert sum(counts) == 43
    # The last rank carries the MTP block, so it must stay well under a full
    # share; it is no longer forced to exactly zero.
    assert counts[-1] < max(counts)
    assert all(counts[0] <= c for c in counts[1:-1])


def test_compute_layer_range_draft_zero_last():
    assert compute_layer_range(43, 12, 0, draft_zero_last=True) == (0, 3)
    assert compute_layer_range(43, 12, 1, draft_zero_last=True) == (3, 7)
    assert compute_layer_range(43, 12, 10, draft_zero_last=True) == (39, 42)
    assert compute_layer_range(43, 12, 11, draft_zero_last=True) == (42, 43)


def test_compute_layer_range():
    assert compute_layer_range(7, 3, 0) == (0, 2)
    assert compute_layer_range(7, 3, 1) == (2, 5)
    assert compute_layer_range(7, 3, 2) == (5, 7)
    assert compute_layer_range(7, 4, 0) == (0, 2)
    assert compute_layer_range(7, 4, 1) == (2, 4)
    assert compute_layer_range(7, 4, 3) == (6, 7)


def test_select_shards_uses_pp_rank_after_tp_grouping(tmp_path):
    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text(json.dumps({"num_hidden_layers": 7}))
    index.write_text(
        json.dumps({
            "weight_map": {
                "embed_tokens.weight": "embed.safetensors",
                "model.layers.0.weight": "l0.safetensors",
                "model.layers.1.weight": "l1.safetensors",
                "model.layers.2.weight": "l2.safetensors",
                "model.layers.3.weight": "l3.safetensors",
                "model.layers.4.weight": "l4.safetensors",
                "model.layers.5.weight": "l5.safetensors",
                "model.layers.6.weight": "l6.safetensors",
                "model.norm.weight": "tail.safetensors",
                "lm_head.weight": "tail.safetensors",
                "mtp.layers.0.weight": "mtp.safetensors",
            }
        }))

    assert select_shards(index, config, rank=0, tp_size=2, pp_size=3) == [
        "embed.safetensors", "l0.safetensors", "l1.safetensors"
    ]
    assert select_shards(index, config, rank=2, tp_size=2, pp_size=3) == [
        "l2.safetensors", "l3.safetensors", "l4.safetensors"
    ]
    assert select_shards(index, config, rank=4, tp_size=2, pp_size=3) == [
        "l5.safetensors", "l6.safetensors", "mtp.safetensors",
        "tail.safetensors"
    ]


def test_select_shards_draft_zero_last_matches_partition(tmp_path):
    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text(json.dumps({"num_hidden_layers": 7}))
    index.write_text(
        json.dumps({
            "weight_map": {
                "embed_tokens.weight": "embed.safetensors",
                **{
                    f"model.layers.{i}.weight": f"l{i}.safetensors"
                    for i in range(7)
                },
                "model.norm.weight": "tail.safetensors",
                "lm_head.weight": "tail.safetensors",
                "mtp.layers.0.weight": "mtp.safetensors",
            }
        }))

    # Shards must follow the same draft partition as the partition subcommand.
    assert compute_layer_counts(7, 3, draft_zero_last=True) == [3, 3, 1]
    assert select_shards(
        index, config, rank=0, tp_size=1, pp_size=3,
        draft_zero_last=True) == [
            "embed.safetensors", "l0.safetensors", "l1.safetensors",
            "l2.safetensors"
        ]
    assert select_shards(
        index, config, rank=1, tp_size=1, pp_size=3,
        draft_zero_last=True) == [
            "l3.safetensors", "l4.safetensors", "l5.safetensors"
        ]
    # Last rank retains the target/draft seam plus the draft stages and head.
    assert select_shards(
        index, config, rank=2, tp_size=1, pp_size=3,
        draft_zero_last=True) == [
            "l6.safetensors", "mtp.safetensors", "tail.safetensors"
        ]


# ---------------------------------------------------------------------------
# MTP-aware cost balancing
# ---------------------------------------------------------------------------


def _dsv4_config(num_nextn_predict_layers=3, num_hidden_layers=DSV4_LAYERS):
    config = {"num_hidden_layers": num_hidden_layers}
    if num_nextn_predict_layers is not None:
        config["num_nextn_predict_layers"] = num_nextn_predict_layers
    return config


def _dsv4_weight_map(stages=3, num_hidden_layers=DSV4_LAYERS, per_layer=10):
    """Weight map shaped like the real index: N layer groups + an mtp group.

    The real checkpoint has 43 layer groups averaging 1571 tensors and 4708
    ``mtp.<stage>.*`` tensors; this keeps the same 3.0x ratio at small scale.
    """
    weight_map = {"embed.weight": "e.safetensors", "head.weight": "t.safetensors"}
    for layer in range(num_hidden_layers):
        for i in range(per_layer):
            weight_map[f"layers.{layer}.w{i}"] = f"l{layer}.safetensors"
    for stage in range(stages):
        for i in range(per_layer):
            weight_map[f"mtp.{stage}.w{i}"] = "mtp.safetensors"
    return weight_map


def _rank_costs(counts, mtp_cost, embed_cost):
    costs = [float(c) for c in counts]
    costs[0] += embed_cost
    costs[-1] += mtp_cost
    return costs


def test_pp12_draft_partition_is_not_valid_at_pp11():
    """A PP=12 partition cannot be truncated for an 11-rank job.

    The truncated string does not sum to the model's layer count, so vLLM must
    reject it rather than silently changing ownership at the MTP seam.
    """
    pp12 = compute_layer_counts(DSV4_LAYERS, 12, mtp_cost=DSV4_MTP_COST)
    chopped = pp12[:-1]
    assert sum(chopped) == DSV4_LAYERS - 1
    assert chopped == [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]

    pp11 = compute_layer_counts(DSV4_LAYERS, 11, mtp_cost=DSV4_MTP_COST)
    assert pp11 != chopped
    # The failing string gave the last rank a full share on top of the MTP.
    assert pp11[-1] < max(pp11)


@pytest.mark.parametrize("pp_size", list(range(2, 17)))
def test_last_rank_never_gets_mtp_plus_a_full_share(pp_size):
    counts = compute_layer_counts(DSV4_LAYERS, pp_size, mtp_cost=DSV4_MTP_COST)
    costs = _rank_costs(counts, DSV4_MTP_COST, partition.DEFAULT_EMBED_COST)
    assert costs[-1] <= max(costs) + 1e-9
    assert counts[-1] + DSV4_MTP_COST <= max(counts) + DSV4_MTP_COST


def test_dsv4_reference_partitions_with_mtp():
    assert compute_layer_counts(DSV4_LAYERS, 11, mtp_cost=DSV4_MTP_COST) == [
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 1
    ]
    # The balancer independently reproduces the validated PP=12 partition.
    assert compute_layer_counts(DSV4_LAYERS, 12, mtp_cost=DSV4_MTP_COST) == [
        3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1
    ]


def test_dsv4_reference_partitions_without_mtp():
    assert compute_layer_counts(DSV4_LAYERS, 11) == [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3
    ]
    assert compute_layer_counts(DSV4_LAYERS, 12) == [
        3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3
    ]


def test_detect_mtp_cost_from_config_num_nextn_predict_layers():
    assert partition.detect_mtp_cost(_dsv4_config(3)) == pytest.approx(3.0)
    assert partition.detect_mtp_cost(_dsv4_config(1)) == pytest.approx(1.0)


def test_detect_mtp_cost_absent_without_signal():
    assert partition.detect_mtp_cost({"num_hidden_layers": 43}) == 0.0
    assert partition.detect_mtp_cost(_dsv4_config(0)) == 0.0
    assert partition.detect_mtp_cost({"num_hidden_layers": 7},
                                     {"layers.0.w": "a", "head.weight": "t"}) == 0.0


def test_detect_mtp_cost_from_weight_index_overrides_declared_count():
    # DeepSeek-V4-Flash-0731 declares num_nextn_predict_layers: 1 while
    # shipping all three DSpark stages; the tensor counts are the truth.
    weight_map = _dsv4_weight_map(stages=3)
    assert partition.detect_mtp_cost(_dsv4_config(1),
                                     weight_map) == pytest.approx(3.0, abs=0.05)
    assert partition.detect_mtp_cost(_dsv4_config(None),
                                     weight_map) == pytest.approx(3.0, abs=0.05)


def test_resolve_mtp_cost_precedence(tmp_path):
    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text(json.dumps(_dsv4_config(3)))
    index.write_text(json.dumps({"weight_map": _dsv4_weight_map(stages=3)}))

    # auto
    assert partition.resolve_mtp_cost(config) == pytest.approx(3.0)
    assert partition.resolve_mtp_cost(config, index) == pytest.approx(3.0, abs=0.05)
    # explicit off wins over a detected checkpoint
    assert partition.resolve_mtp_cost(config, index, draft_zero_last=False) == 0.0
    # explicit on wins over a checkpoint with no signal
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"num_hidden_layers": 43}))
    assert partition.resolve_mtp_cost(plain) == 0.0
    assert partition.resolve_mtp_cost(
        plain, draft_zero_last=True) == pytest.approx(partition.DEFAULT_MTP_COST)


def test_explicit_draft_zero_last_flag_still_forces_mtp_mode():
    flagged = compute_layer_counts(DSV4_LAYERS, 11, draft_zero_last=True)
    detected = compute_layer_counts(DSV4_LAYERS, 11,
                                    mtp_cost=partition.DEFAULT_MTP_COST)
    assert flagged == detected
    assert flagged != compute_layer_counts(DSV4_LAYERS, 11)


def test_explicit_mtp_cost_zero_forces_legacy_split():
    assert compute_layer_counts(DSV4_LAYERS, 11, mtp_cost=0.0) == \
        compute_layer_counts(DSV4_LAYERS, 11)


@pytest.mark.parametrize("pp_size", list(range(2, 17)))
@pytest.mark.parametrize("num_layers", [0, 1, 7, 43, 61, 64])
def test_sum_and_non_negative_invariants(num_layers, pp_size):
    for mtp_cost in (0.0, 1.0, DSV4_MTP_COST):
        counts = compute_layer_counts(num_layers, pp_size, mtp_cost=mtp_cost)
        assert len(counts) == pp_size
        assert sum(counts) == num_layers
        assert all(c >= 0 for c in counts)


@pytest.mark.parametrize("pp_size", list(range(2, 17)))
@pytest.mark.parametrize("num_layers", [7, 43, 61, 64])
def test_cost_balance_no_rank_more_than_one_layer_over_the_lightest(
        num_layers, pp_size):
    counts = compute_layer_counts(num_layers, pp_size, mtp_cost=DSV4_MTP_COST)
    costs = _rank_costs(counts, DSV4_MTP_COST, partition.DEFAULT_EMBED_COST)
    if min(counts) >= 1:
        assert max(costs) - min(costs) <= 1.0 + 1e-9


@pytest.mark.parametrize("pp_size", list(range(2, 17)))
def test_cost_balance_beats_zero_last(pp_size):
    """The balancer's peak rank is never heavier than the zero-last policy's."""
    balanced = compute_layer_counts(DSV4_LAYERS, pp_size,
                                    mtp_cost=DSV4_MTP_COST)
    zero_last = partition._legacy_draft_zero_last_counts(DSV4_LAYERS, pp_size)
    embed = partition.DEFAULT_EMBED_COST
    assert max(_rank_costs(balanced, DSV4_MTP_COST, embed)) <= \
        max(_rank_costs(zero_last, DSV4_MTP_COST, embed)) + 1e-9


def test_pp11_peak_rank_is_lighter_than_zero_last():
    balanced = compute_layer_counts(DSV4_LAYERS, 11, mtp_cost=DSV4_MTP_COST)
    assert max(balanced) == 5
    assert balanced[-1] == 1
    # zero-last leaves the MTP rank at 3.0 while three ranks carry 5 layers.
    assert partition._legacy_draft_zero_last_counts(DSV4_LAYERS, 11) == [
        4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 0
    ]


def test_edge_cases():
    assert compute_layer_counts(0, 4, mtp_cost=DSV4_MTP_COST) == [0, 0, 0, 0]
    # pp_size > num_layers: the MTP rank is already the most expensive, so it
    # takes no target layers and the rest load from the tail.
    assert compute_layer_counts(2, 5, mtp_cost=DSV4_MTP_COST) == [0, 0, 1, 1, 0]
    assert compute_layer_counts(43, 1, mtp_cost=DSV4_MTP_COST) == [43]
    with pytest.raises(ValueError):
        compute_layer_counts(-1, 4)
    with pytest.raises(ValueError):
        compute_layer_counts(43, 0)
    # Preserved: the explicit flag has always required pp_size >= 2.
    with pytest.raises(ValueError):
        compute_layer_counts(43, 1, draft_zero_last=True)


def test_compute_layer_range_tracks_the_balanced_counts():
    counts = compute_layer_counts(DSV4_LAYERS, 11, mtp_cost=DSV4_MTP_COST)
    start = 0
    for pp_rank, count in enumerate(counts):
        assert compute_layer_range(DSV4_LAYERS, 11, pp_rank,
                                   mtp_cost=DSV4_MTP_COST) == (start,
                                                               start + count)
        start += count
    assert start == DSV4_LAYERS


def test_select_shards_auto_detects_mtp(tmp_path):
    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text(json.dumps({
        "num_hidden_layers": 7,
        "num_nextn_predict_layers": 3
    }))
    index.write_text(
        json.dumps({
            "weight_map": {
                "embed_tokens.weight": "embed.safetensors",
                **{f"model.layers.{i}.weight": f"l{i}.safetensors"
                   for i in range(7)},
                "model.norm.weight": "tail.safetensors",
                "lm_head.weight": "tail.safetensors",
                **{f"mtp.{s}.weight": "mtp.safetensors" for s in range(3)},
            }
        }))

    # Auto-detection alone reproduces the draft partition: 7 layers, pp=3.
    counts = compute_layer_counts(7, 3, mtp_cost=3.0)
    assert counts == [3, 3, 1]
    assert select_shards(index, config, rank=2, tp_size=1,
                         pp_size=3) == [
                             "l6.safetensors", "mtp.safetensors",
                             "tail.safetensors"
                         ]
    # Explicitly off falls back to the plain split: last rank owns layers 5-6.
    assert select_shards(index, config, rank=2, tp_size=1, pp_size=3,
                         draft_zero_last=False) == [
                             "l5.safetensors", "l6.safetensors",
                             "mtp.safetensors", "tail.safetensors"
                         ]


def _cli(*args, cwd):
    result = subprocess.run([sys.executable, str(MODULE_PATH), *args],
                            capture_output=True,
                            text=True,
                            check=True)
    return result.stdout.strip()


def test_cli_partition_auto_detect_and_overrides(tmp_path):
    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text(json.dumps(_dsv4_config(3)))
    index.write_text(json.dumps({"weight_map": _dsv4_weight_map(stages=3)}))

    auto = _cli("partition", "--config", str(config), "--pp-size", "11",
                cwd=tmp_path)
    assert auto == "4,4,4,4,4,4,4,4,5,5,1"

    with_index = _cli("partition", "--config", str(config), "--index",
                      str(index), "--pp-size", "11", cwd=tmp_path)
    assert with_index == auto

    forced_on = _cli("partition", "--config", str(config), "--pp-size", "11",
                     "--draft-zero-last", cwd=tmp_path)
    assert forced_on == auto

    forced_off = _cli("partition", "--config", str(config), "--pp-size", "11",
                      "--no-draft-zero-last", cwd=tmp_path)
    assert forced_off == "4,4,4,4,4,4,4,4,4,4,3"


def test_cli_layers_auto_detect(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_dsv4_config(3)))
    assert _cli("layers", "--config", str(config), "--pp-size", "11", "--rank",
                "10", cwd=tmp_path) == "42:43"
    assert _cli("layers", "--config", str(config), "--pp-size", "11", "--rank",
                "10", "--no-draft-zero-last", cwd=tmp_path) == "40:43"


def test_weight_index_cost_is_quantized_so_the_partition_is_stable(tmp_path):
    """Real ratio 4708/1571.23 = 2.9964 must not undercut a 3.0 stage count.

    Left raw it makes the draft rank the cheapest by 0.004 and it wins every
    tie, pulling an extra layer onto the one rank that must stay light.
    """
    # 2997 mtp tensors over a 1000-tensor mean layer = 2.997 raw.
    weight_map = {"embed.weight": "e"}
    for layer in range(43):
        for i in range(1000):
            weight_map[f"layers.{layer}.w{i}"] = f"l{layer}"
    for i in range(2997):
        weight_map[f"mtp.{i % 3}.w{i}"] = "mtp"

    assert partition.detect_mtp_cost(_dsv4_config(3), weight_map) == 3.0
    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text(json.dumps(_dsv4_config(3)))
    index.write_text(json.dumps({"weight_map": weight_map}))
    assert partition.resolve_mtp_cost(config, index) == \
        partition.resolve_mtp_cost(config)
    for pp_size in range(2, 17):
        assert compute_layer_counts(
            DSV4_LAYERS, pp_size,
            mtp_cost=partition.resolve_mtp_cost(config, index)) == \
            compute_layer_counts(DSV4_LAYERS, pp_size,
                                 mtp_cost=partition.resolve_mtp_cost(config))
