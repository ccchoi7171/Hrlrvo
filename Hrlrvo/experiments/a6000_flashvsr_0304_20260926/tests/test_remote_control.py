from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


remote = load("test_remote_control_module", "tools/remote_control.py")
worker = load("test_remote_control_tile_worker", "tools/tile_worker.py")
composite = load(
    "test_remote_control_flash_composite",
    "package/src/cert_pipeline/flash_composite.py",
)


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


@pytest.mark.parametrize(
    ("gpu", "video", "start", "end"),
    [(0, "04", 0, 84), (1, "04", 85, 169),
     (2, "03", 0, 84), (3, "03", 85, 169)],
)
def test_full_worker_assignment_and_input_anchor_mapping(gpu, video, start, end):
    command, work = remote.worker_command(gpu, "full")
    assert Path(command[1]).name == "tile_worker.py"
    assert option(command, "--video-id") == video
    assert int(option(command, "--gpu-index")) == gpu
    assert int(option(command, "--tile-start")) == start
    assert int(option(command, "--tile-end")) == end
    assert option(command, "--mode") == "full"
    assert Path(option(command, "--input")).name == f"{video}_Outpaint_4K.mp4"
    assert Path(option(command, "--anchor")).name == f"{video}_Input_8K.mp4"
    assert work == remote.paths(video)["work"] / f"gpu{gpu}"


def test_full_worker_delegates_fixed_256_32_contract():
    command, _ = remote.worker_command(0, "full")
    assert Path(command[1]).resolve() == (remote.ROOT / "tools/tile_worker.py").resolve()
    assert worker.TILE_SIZE == 256
    assert worker.TILE_OVERLAP == 32
    assert worker.BASE_SEED == 71634910
    assert worker.SPARSE_RATIO == 2.0
    assert worker.LOCAL_RANGE == 11


def test_bad_source_provenance_sha_fails_before_asset_reads(monkeypatch):
    monkeypatch.setattr(remote, "digest", lambda _path: "0" * 64)
    monkeypatch.setattr(
        remote,
        "read",
        lambda _path: pytest.fail("provenance content must not be trusted after bad SHA"),
    )
    with pytest.raises(RuntimeError, match="provenance manifest changed"):
        remote.verify_assets("a" * 64, "b" * 64)


def test_verified_asset_stat_change_gate_fails(tmp_path, monkeypatch):
    proof = {
        "status": "PASS",
        "source_manifest_sha256": remote.SOURCE_SHA,
        "manifests": {},
        "files": {
            "package/asset.bin": {
                "identity": {"size": 1, "mtime_ns": 1, "ctime_ns": 1,
                             "inode": 1, "device": 1},
                "sha256": "a" * 64,
            }
        },
    }
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    monkeypatch.setattr(remote, "read", lambda _path: proof)
    monkeypatch.setattr(remote, "inside", lambda path, root=None: Path(path).resolve())
    monkeypatch.setattr(
        remote,
        "identity",
        lambda _path: {"size": 1, "mtime_ns": 2, "ctime_ns": 1,
                       "inode": 1, "device": 1},
    )
    with pytest.raises(RuntimeError, match="Verified asset changed"):
        remote.check_assets()


def test_full_launch_cannot_reach_tmux_without_compatibility_pass(monkeypatch):
    def reject_gate(*, smoke=False):
        assert smoke is True
        raise RuntimeError("compatibility PASS missing")

    monkeypatch.setattr(remote, "check_gate", reject_gate)
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("tmux probe must not run before gate"),
    )
    monkeypatch.setattr(
        remote,
        "run",
        lambda *_args, **_kwargs: pytest.fail("launch command must not run before gate"),
    )
    with pytest.raises(RuntimeError, match="compatibility PASS missing"):
        remote.launch_full()


@pytest.mark.parametrize("existing", ["work", "raw", "final", "manifest"])
def test_full_launch_fresh_guard_rejects_existing_outputs(
    tmp_path, monkeypatch, existing
):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    monkeypatch.setattr(remote, "PYTHON", tmp_path / "env/bin/python")
    monkeypatch.setattr(remote, "check_gate", lambda *, smoke=False: None)
    monkeypatch.setattr(
        remote.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=2**41, used=0, free=2**41),
    )
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(remote, "run", lambda command, **_kwargs: launched.append(command))
    target = remote.paths("04")[existing]
    if existing == "work":
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="workspace not fresh"):
        remote.launch_full()
    assert launched == []


@pytest.mark.parametrize("video", ["03", "04"])
def test_postprocess_command_is_accepted_by_postprocess_parser(video):
    command = remote.postprocess_command(video)
    assert Path(command[1]).name == "postprocess_video.py"
    parser = composite.parser()
    parser.add_argument("--aggregate-manifest", type=Path, required=True)
    parsed = parser.parse_args(command[2:])
    assert parsed.video_id == video
    assert str(parsed.tile_size) == "256"
    assert str(parsed.tile_overlap) == "32"
    assert parsed.expected_frames == 1952
    assert parsed.aggregate_manifest.name == "aggregate_manifest.json"


def test_wrong_video_is_rejected_without_side_effects():
    with pytest.raises(RuntimeError, match="Invalid video ID"):
        remote.paths("05")
    with pytest.raises(RuntimeError, match="Invalid video ID"):
        remote.postprocess_command("05")


@pytest.mark.parametrize(
    "workers",
    [
        [{"gpu": gpu} for gpu in range(3)],
        [{"gpu": gpu} for gpu in (0, 1, 2, 2)],
    ],
)
def test_compatibility_pass_rejects_missing_or_duplicate_gpus(monkeypatch, workers):
    environment = {
        "status": "PASS",
        "transfer_receipt_sha256": "transfer-hash",
        "all_package_versions": {"torch": "pinned"},
        "environment_file_identities": {"bin/python": {"size": 1}},
    }
    compatibility = {
        "status": "PASS",
        "environment_receipt_sha256": "environment-hash",
        "workers": workers,
    }

    def fake_read(path):
        return compatibility if Path(path).name == "compatibility_test.json" else environment

    def fake_digest(path):
        return (
            "environment-hash"
            if Path(path).name == "environment_preflight.json"
            else "transfer-hash"
        )

    monkeypatch.setattr(remote, "check_assets", lambda: {})
    monkeypatch.setattr(remote, "read", fake_read)
    monkeypatch.setattr(remote, "digest", fake_digest)
    monkeypatch.setattr(remote, "installed_versions", lambda: {"torch": "pinned"})
    monkeypatch.setattr(
        remote, "environment_identities", lambda: {"bin/python": {"size": 1}}
    )
    with pytest.raises(RuntimeError, match="exactly GPUs 0,1,2,3"):
        remote.check_gate(smoke=True)


def test_caller_manifest_digest_mismatch_fails_before_manifest_read(monkeypatch):
    source = {
        "upstream_model_commit": remote.MODEL_COMMIT,
        "block_sparse_commit": remote.BSA_COMMIT,
        "files": [],
    }

    def fake_digest(path):
        name = Path(path).name
        if name == "source_provenance.json":
            return remote.SOURCE_SHA
        if name == "code_manifest.json":
            return "0" * 64
        return "f" * 64

    def fake_read(path):
        name = Path(path).name
        if name == "source_provenance.json":
            return source
        pytest.fail(f"transferred manifest must not be read after digest mismatch: {name}")

    monkeypatch.setattr(remote, "digest", fake_digest)
    monkeypatch.setattr(remote, "read", fake_read)
    with pytest.raises(RuntimeError, match="differs from local approval: code_manifest"):
        remote.verify_assets("a" * 64, "b" * 64)


def test_await_barrier_rejects_wrong_release_token(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    status = tmp_path / "status"
    status.mkdir()
    remote.write(status / "launch_reservation.json", {"token": "expected"})
    remote.write(status / "start_barrier.json", {"token": "wrong"})
    monkeypatch.setattr(remote, "check_gate", lambda *, smoke=False: None)
    monkeypatch.setattr(remote.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        remote.time,
        "sleep",
        lambda _seconds: pytest.fail("existing barrier must not require waiting"),
    )
    with pytest.raises(RuntimeError, match="Wrong start barrier token"):
        remote.await_start_barrier("supervise", 0, "expected")


def test_launcher_releases_barrier_only_after_all_six_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    monkeypatch.setattr(remote, "PYTHON", tmp_path / "env/bin/python")
    monkeypatch.setattr(remote, "check_gate", lambda *, smoke=False: None)
    monkeypatch.setattr(remote.secrets, "token_hex", lambda _length: "fixed-token")
    monkeypatch.setattr(remote.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        remote.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=2**41, used=0, free=2**41),
    )
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(remote, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote, "digest", lambda _path: "compatibility-hash")

    status = tmp_path / "status"
    status.mkdir(parents=True)
    for gpu in range(4):
        remote.write(status / f"gpu{gpu}.json", {"status": "COMPLETE"})
    ready_names = [
        "ready_supervise_0.json", "ready_supervise_1.json",
        "ready_supervise_2.json", "ready_supervise_3.json",
        "ready_coordinate_04.json", "ready_coordinate_03.json",
    ]
    sleep_calls = 0

    def release_one_ready(_seconds):
        nonlocal sleep_calls
        assert not (status / "start_barrier.json").exists()
        remote.write(status / ready_names[sleep_calls], {"token": "fixed-token"})
        sleep_calls += 1

    monkeypatch.setattr(remote.time, "sleep", release_one_ready)
    remote.launch_full()
    assert sleep_calls == 6
    assert all((status / name).is_file() for name in ready_names)
    assert remote.read(status / "start_barrier.json")["token"] == "fixed-token"


@pytest.mark.parametrize('video', ['03', '04'])
def test_raw_target_satisfies_merge_work_directory_containment(video):
    spec = remote.paths(video)
    assert spec['raw'].is_relative_to(spec['work'])


def test_direct_weight_move_waits_for_rsync_checkpoint_directory():
    script = (ROOT / 'tools/remote_prepare.sh').read_text()
    guard = script.index('[ -d "$D/checkpoint" ]')
    move = script.index('mv -n "$D/FlashVSR-v1.1/')
    assert guard < move


def test_editable_relocation_changes_only_isolated_metadata(tmp_path, monkeypatch):
    script = (ROOT / 'tools/remote_prepare.sh').read_text()
    snippet = script.split("<<'RELOCATE_PY'\n", 1)[1].split('\nRELOCATE_PY', 1)[0]
    site = tmp_path / 'env/lib/python3.11/site-packages'
    site.mkdir(parents=True)
    old = '/home/work/ai-graphics_33/Hrlrvo/Upscaler/models/FlashVSR_v1.1/repository'
    for name in ('diffsynth.egg-link', 'easy-install.pth'):
        (site / name).write_text(old + '\n')
    monkeypatch.setattr(sys, 'argv', ['-', str(tmp_path)])
    exec(compile(snippet, '<editable-relocation-test>', 'exec'), {})
    expected = str(tmp_path / 'package/src/models/flashvsr_v1_1/repository') + '\n'
    for name in ('diffsynth.egg-link', 'easy-install.pth'):
        assert (site / name).read_text() == expected
    assert not (tmp_path / 'package').exists()


def test_gpu_uuid_format_normalization_does_not_change_device_identity():
    gpu0 = remote.GPU_UUIDS[0]
    assert remote.same_gpu_uuid(gpu0, gpu0)
    assert remote.same_gpu_uuid(gpu0.removeprefix('GPU-'), gpu0)
    assert not remote.same_gpu_uuid(remote.GPU_UUIDS[1], gpu0)
    assert not remote.same_gpu_uuid('prefix-' + gpu0, gpu0)
