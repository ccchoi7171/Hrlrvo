import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_video02_assignments_are_exact_complement():
    worker = load("tools/tile_worker.py", "test_video02_worker")
    values = []
    for gpu in range(4):
        start, end = worker.ASSIGNMENTS[("02", gpu)]
        values.extend(range(start, end + 1))
    assert values == list(range(170))
    assert worker.BASE_SEED == 25621683
    assert worker.TILE_SIZE == 256
    assert worker.TILE_OVERLAP == 32
    assert worker.SPARSE_RATIO == 2.0
    assert worker.LOCAL_RANGE == 11


def test_merge_requires_four_and_same_seed():
    merge = load("tools/merge_shards.py", "test_video02_merge")
    assert merge.BASE_SEED == 25621683
    assert set(merge.ASSIGNMENTS) == {("02", gpu) for gpu in range(4)}
    assert [i for gpu in range(4) for i in merge.ASSIGNMENTS[("02", gpu)]] == list(range(170))


def test_orchestrator_contract_and_commands():
    orchestrator = load("tools/orchestrate.py", "test_video02_orchestrator")
    assert orchestrator.BASE_SEED == 25621683
    assert orchestrator.ASSIGNMENTS == {0:(0,42),1:(43,85),2:(86,127),3:(128,169)}
    assert "--seed" in orchestrator.final_command()
    assert orchestrator.final_command()[orchestrator.final_command().index("--seed")+1] == "25621683"
    for gpu in range(4):
        command = orchestrator.worker_command(gpu)
        assert command[command.index("--tile-start")+1] == str(orchestrator.ASSIGNMENTS[gpu][0])
        assert command[command.index("--tile-end")+1] == str(orchestrator.ASSIGNMENTS[gpu][1])
