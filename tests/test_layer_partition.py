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
