import json
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "vllm" / "layer_partition.py"
SPEC = importlib.util.spec_from_file_location("layer_partition", MODULE_PATH)
assert SPEC and SPEC.loader
partition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partition)

compute_layer_counts = partition.compute_layer_counts
compute_layer_range = partition.compute_layer_range
select_shards = partition.select_shards


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
    # Validated hand partition for the DSV4 int4/int8 + DSpark checkpoint
    # (43 target layers, 3 grafted draft stages on the last rank).
    assert compute_layer_counts(43, 12, draft_zero_last=True) == [
        3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0
    ]


def test_compute_layer_counts_draft_zero_last_pp10_properties():
    counts = compute_layer_counts(43, 10, draft_zero_last=True)
    assert sum(counts) == 43
    assert counts[-1] == 0
    assert all(counts[0] <= c for c in counts[1:-1])


def test_compute_layer_range_draft_zero_last():
    assert compute_layer_range(43, 12, 0, draft_zero_last=True) == (0, 3)
    assert compute_layer_range(43, 12, 1, draft_zero_last=True) == (3, 7)
    assert compute_layer_range(43, 12, 10, draft_zero_last=True) == (39, 43)
    assert compute_layer_range(43, 12, 11, draft_zero_last=True) == (43, 43)


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

    # shards must follow the same draft partition as the partition
    # subcommand: 7 layers over pp_size=3 -> 3,4,0.
    assert compute_layer_counts(7, 3, draft_zero_last=True) == [3, 4, 0]
    assert select_shards(
        index, config, rank=0, tp_size=1, pp_size=3,
        draft_zero_last=True) == [
            "embed.safetensors", "l0.safetensors", "l1.safetensors",
            "l2.safetensors"
        ]
    assert select_shards(
        index, config, rank=1, tp_size=1, pp_size=3,
        draft_zero_last=True) == [
            "l3.safetensors", "l4.safetensors", "l5.safetensors",
            "l6.safetensors"
        ]
    # Last rank owns zero target layers: only the draft stages and head.
    assert select_shards(
        index, config, rank=2, tp_size=1, pp_size=3,
        draft_zero_last=True) == ["mtp.safetensors", "tail.safetensors"]
