from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ltx_outpaint.train import OutpaintingTrainer, load_training_config
from ltx_outpaint.training_runtime import (
    config_fingerprint,
    prepare_run_artifacts,
    run_training_with_health,
)
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path):
    config, outpainting = load_training_config(ROOT / "configs/train_4k.yaml")
    config.output_dir = str(tmp_path / "run")
    return config, outpainting


def test_config_fingerprint_ignores_only_resume_path_and_catches_horizon_and_lr(
    tmp_path,
):
    config, outpainting = _config(tmp_path)
    baseline = config_fingerprint(config, outpainting)

    resumed = copy.deepcopy(config)
    resumed.model.load_checkpoint = "/different/checkpoint-000005"
    assert config_fingerprint(resumed, outpainting) == baseline

    changed_horizon = copy.deepcopy(config)
    changed_horizon.optimization.steps += 1
    assert config_fingerprint(changed_horizon, outpainting) != baseline

    changed_lr = copy.deepcopy(config)
    changed_lr.optimization.learning_rate = 2e-4
    assert config_fingerprint(changed_lr, outpainting) != baseline


@pytest.mark.parametrize("horizon", [2000, 30000])
def test_native_linear_scheduler_preserves_kwargs_and_configured_horizon(tmp_path, horizon):
    config, _ = _config(tmp_path)
    config.optimization.steps = horizon
    trainer = OutpaintingTrainer.__new__(OutpaintingTrainer)
    trainer._config = config
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=config.optimization.learning_rate)
    before = copy.deepcopy(config.optimization.scheduler_params)

    scheduler = trainer._create_scheduler(optimizer)

    assert config.optimization.steps == horizon
    assert config.optimization.scheduler_type == "linear"
    assert config.optimization.scheduler_params == before == {
        "start_factor": 1.0,
        "end_factor": 0.1,
    }
    assert scheduler.total_iters == horizon
    assert scheduler.start_factor == 1.0
    assert scheduler.end_factor == 0.1


def test_prepare_run_artifacts_never_overwrites_and_resume_uses_original_dir(
    tmp_path,
):
    config, outpainting = _config(tmp_path)
    manifest = prepare_run_artifacts(config, outpainting, None)
    manifest_path = Path(config.output_dir) / "run_manifest.json"
    original = manifest_path.read_bytes()
    assert manifest["scheduler_horizon_optimizer_steps"] == config.optimization.steps
    assert (Path(config.output_dir) / "run_config.yaml").is_file()

    with pytest.raises(FileExistsError, match="Run already exists"):
        prepare_run_artifacts(config, outpainting, None)

    resumed = copy.deepcopy(config)
    checkpoint = Path(config.output_dir) / "checkpoints" / "checkpoint-000005"
    resumed.model.load_checkpoint = str(checkpoint)
    assert prepare_run_artifacts(resumed, outpainting, checkpoint) == manifest
    assert manifest_path.read_bytes() == original
    assert (Path(config.output_dir) / "resume_config_checkpoint-000005.yaml").is_file()

    wrong_root = copy.deepcopy(resumed)
    wrong_root.output_dir = str(tmp_path / "different-run")
    with pytest.raises(FileNotFoundError, match="original run_manifest"):
        prepare_run_artifacts(wrong_root, outpainting, checkpoint)


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2), requires_grad=False)


class _FrozenModel(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn1 = nn.Module()
        block.attn1.to_q = _Projection()
        block.attn1.to_out = nn.Sequential(_Projection())
        block.ff = nn.Module()
        block.ff.net = nn.ModuleList([nn.Module(), nn.Identity(), nn.Identity()])
        block.ff.net[0].proj = _Projection()
        block.attn2 = nn.Module()
        block.attn2.to_q = _Projection()
        self.transformer_blocks = nn.ModuleList([block])
        self.runtime_trainable = nn.Parameter(torch.ones(()))


class _OptimizerBox:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


class _FakeHealth:
    def __init__(self, *_args, **_kwargs):
        pass

    def begin_optimizer_step(self):
        pass

    def begin_forward(self):
        pass

    def end_forward(self):
        pass

    def set_geometry(self, *_args, **_kwargs):
        pass

    def capture_pre_clip_gradients(self):
        return {"grad_norm_global": 1.0}

    def capture_post_clip_global_norm(self):
        return 1.0

    def snapshot_representatives(self):
        return {}

    def representative_updates(self, _snapshots):
        return {"runtime_trainable": True}

    def fusion_metrics(self):
        return {}

    def log_step(self, _row):
        pass

    def close(self):
        pass


class _FakeAccelerator:
    num_processes = 1
    is_main_process = True
    device = torch.device("cpu")
    step = 0

    def unwrap_model(self, model, **_kwargs):
        return model

    def clip_grad_norm_(self, parameters, max_norm, *args, **kwargs):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, *args, **kwargs)

    def wait_for_everyone(self):
        pass

    def end_training(self):
        pass


class _FakeStrategy:
    def prepare_training_inputs(self, _batch, _sampler):
        loss_mask = torch.tensor([[False, True]])
        return SimpleNamespace(
            video_loss_mask=loss_mask,
            video=SimpleNamespace(
                timesteps=torch.tensor([[0.0, 0.5]]),
                geometry=SimpleNamespace(),
                latent=torch.zeros(1, 2),
            ),
            audio=None,
        )


class _FakeTrainer:
    def __init__(self, root: Path, initial_step: int):
        self._transformer = _FrozenModel()
        self._accelerator = _FakeAccelerator()
        self._training_strategy = _FakeStrategy()
        self._resume_state = (initial_step, None)
        self._global_step = initial_step
        self._runtime_state = {
            "run_id": root.name,
            "optimizer_step": initial_step,
            "batches_seen": initial_step,
            "samples_seen": initial_step,
        }
        self._config = SimpleNamespace(
            output_dir=str(root),
            optimization=SimpleNamespace(
                steps=2000,
                batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
            ),
        )
        self._parameter_counts = {"trainable_total": 1}
        self._timestep_sampler = object()
        self._lr_scheduler = SimpleNamespace(state_dict=lambda: {"last_epoch": initial_step})

    def _init_optimizer(self):
        optimizer = torch.optim.SGD(
            [self._transformer.runtime_trainable], lr=1e-4
        )
        optimizer.param_groups[0]["initial_lr"] = 1e-4
        self._optimizer = _OptimizerBox(optimizer)

    def _training_step(self, batch):
        self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)
        self._runtime_state["batches_seen"] += 1
        self._runtime_state["samples_seen"] += batch["video_latents"]["latents"].shape[0] * self._accelerator.num_processes
        return SimpleNamespace(loss=torch.tensor([1.0]), sigma=torch.tensor([0.5]))

    def _save_checkpoint(self):
        checkpoint = (
            Path(self._config.output_dir)
            / "checkpoints"
            / f"checkpoint-{self._global_step:06d}"
        )
        checkpoint.mkdir(parents=True, exist_ok=True)
        return checkpoint

    def train(self, *, disable_progress_bars, step_callback):
        del disable_progress_bars
        self._init_optimizer()
        batch = {
            "video_latents": {"latents": torch.zeros(1, 1)},
            "anchor_latents": {"latents": torch.zeros(1, 1)},
            "anchor_provenance": [
                {
                    "canonical_latent_rect_yxyx": [0, 0, 1, 1],
                    "anchor_latent_hw": [1, 1],
                    "gt_record_id": "fake.pt",
                }
            ],
            "conditions": {"video_prompt_embeds": torch.zeros(1)},
        }
        for step in range(self._resume_state[0] + 1, self._config.optimization.steps + 1):
            self._training_step(batch)
            self._transformer.runtime_trainable.grad = torch.ones_like(
                self._transformer.runtime_trainable
            )
            self._accelerator.clip_grad_norm_(
                [self._transformer.runtime_trainable], 1.0
            )
            self._optimizer.step()
            self._optimizer.zero_grad()
            self._global_step = step
            step_callback(step, self._config.optimization.steps, [])
        raise AssertionError("bounded callback did not stop the fake native loop")


def test_bounded_runtime_stop_5_then_resume_to_6(monkeypatch, tmp_path):
    from ltx_outpaint import health_logging

    monkeypatch.setattr(health_logging, "TrainingHealthLogger", _FakeHealth)
    root = tmp_path / "runtime"
    root.mkdir()
    first = run_training_with_health(
        _FakeTrainer(root, 0), disable_progress_bars=True, stop_after_step=5
    )
    assert first["initial_step"] == 0
    assert first["final_step"] == 5
    assert first["runtime_state"]["optimizer_step"] == 5
    assert first["runtime_state"]["samples_seen"] == 5
    assert Path(first["checkpoint"]).name == "checkpoint-000005"
    assert len(first["steps"]) == 5

    resumed = run_training_with_health(
        _FakeTrainer(root, 5), disable_progress_bars=True, stop_after_step=6
    )
    assert resumed["initial_step"] == 5
    assert resumed["final_step"] == 6
    assert resumed["runtime_state"]["optimizer_step"] == 6
    assert resumed["runtime_state"]["samples_seen"] == 6
    assert Path(resumed["checkpoint"]).name == "checkpoint-000006"
    assert len(resumed["steps"]) == 1


def test_production_checkpoint_sample_semantics(tmp_path):
    config, _ = _config(tmp_path)
    world_size = 1
    samples = (
        config.checkpoints.interval
        * config.optimization.gradient_accumulation_steps
        * config.optimization.batch_size
        * world_size
    )
    assert config.checkpoints.interval == 500
    assert samples == 500


def test_production_schedule_matches_requested_30000_step_budget(tmp_path):
    import yaml

    config, outpainting = _config(tmp_path)
    # Production cache metadata/file-ID audit on 2026-09-11: 30,024 compatible records.
    cached_records = 30_024
    world_size = 1
    effective_batch = (
        config.optimization.batch_size
        * config.optimization.gradient_accumulation_steps
        * world_size
    )
    assert effective_batch == 1
    assert config.optimization.steps == 30_000
    assert cached_records - config.optimization.steps * effective_batch == 24
    assert config.optimization.learning_rate == 1e-4
    assert outpainting["temporal_sampling"] == "first"
    assert outpainting["clip_frames"] == 65
    inference = yaml.safe_load((ROOT / "configs/inference_4k.yaml").read_text())
    assert Path(inference["model"]["adapter_path"]).parent.name == (
        f"checkpoint-{config.optimization.steps:06d}"
    )


def test_frozen_representative_indices_stay_inside_real_large_weights():
    from ltx_outpaint.training_runtime import _representative_indices

    for numel in (1, 16, 4096 * 4096, 4096 * 16384, 2**31):
        indices = _representative_indices(numel, torch.device('cpu'))
        assert indices.dtype == torch.int64
        assert indices.shape == (32,)
        assert indices.min() == 0
        assert indices.max() == numel - 1
        assert ((0 <= indices) & (indices < numel)).all()
    # The former diagnostic's FP32 endpoint is demonstrably invalid for FFN.
    old = torch.linspace(0, 4096 * 16384 - 1, 32).long()
    assert old[-1] == 4096 * 16384


def test_production_runner_does_not_install_or_compute_diagnostics(monkeypatch, tmp_path, capsys):
    from ltx_outpaint import health_logging, training_runtime

    def forbidden(*_args, **_kwargs):
        pytest.fail("Production must not enter diagnostics")

    monkeypatch.setattr(training_runtime, "run_training_with_health", forbidden)
    monkeypatch.setattr(training_runtime, "_memory", forbidden)
    monkeypatch.setattr(health_logging, "TrainingHealthLogger", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    original_step, original_clip, original_prepare, original_optimizer = (object() for _ in range(4))
    result = object()
    trainer = SimpleNamespace(
        _training_step=original_step,
        _init_optimizer=original_optimizer,
        _training_strategy=SimpleNamespace(prepare_training_inputs=original_prepare),
        _accelerator=SimpleNamespace(num_processes=4, clip_grad_norm_=original_clip),
        _config=SimpleNamespace(optimization=SimpleNamespace(batch_size=2, gradient_accumulation_steps=3)),
        _runtime_state={},
    )

    def native_train(**kwargs):
        assert kwargs == {"disable_progress_bars": True}
        assert trainer._training_step is original_step
        assert trainer._init_optimizer is original_optimizer
        assert trainer._training_strategy.prepare_training_inputs is original_prepare
        assert trainer._accelerator.clip_grad_norm_ is original_clip
        return result

    trainer.train = native_train
    assert training_runtime.run_training(trainer, disable_progress_bars=True) is result
    assert trainer._runtime_state == {"world_size": 4, "effective_batch_size": 24}
    assert not list(tmp_path.iterdir())
    assert "OPTIMIZER_STEP" not in capsys.readouterr().out


@pytest.mark.parametrize("diagnostics", [False, True])
def test_training_cli_diagnostics_are_explicit_opt_in(monkeypatch, tmp_path, diagnostics):
    import contextlib
    import sys

    from ltx_outpaint import train

    config, outpainting = _config(tmp_path)
    calls = []
    trainer = object()

    @contextlib.contextmanager
    def console(_path):
        calls.append("console")
        yield

    monkeypatch.setattr(sys, "argv", ["train", "config.yaml"] + (["--diagnostics"] if diagnostics else []))
    monkeypatch.setattr(train, "load_training_config", lambda _path: (config, outpainting))
    monkeypatch.setattr(train, "prepare_run_artifacts", lambda *_args: None)
    monkeypatch.setattr(train, "training_console", console)
    monkeypatch.setattr(train, "resolve_video_vae_path", lambda *_args: None)
    monkeypatch.setattr(train, "read_video_scale_factors", lambda *_args: None)
    monkeypatch.setattr(train, "audit_ltx_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "_build_anchor_cache", lambda *_args: None)
    monkeypatch.setattr(train, "OutpaintingTrainer", lambda *_args, **_kwargs: trainer)

    def run(actual, **kwargs):
        assert actual is trainer
        assert kwargs == {
            "disable_progress_bars": False, "stop_after_step": None,
            "diagnostics_enabled": diagnostics,
        }
        calls.append("run")

    monkeypatch.setattr(train, "run_training", run)
    train.main()
    assert calls == (["console", "run"] if diagnostics else ["run"])


def test_common_training_step_preserves_resume_counters_without_diagnostics(monkeypatch):
    from ltx_outpaint import train

    trainer = OutpaintingTrainer.__new__(OutpaintingTrainer)
    trainer._accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=4)
    trainer._anchor_cache = SimpleNamespace(complete_batch=lambda _batch: None)
    trainer._constant_condition = {
        key: torch.zeros(1) for key in ("video_prompt_embeds", "audio_prompt_embeds", "prompt_attention_mask")
    }
    # Resume after two accumulation=3, batch=2, world=4 optimizer steps.
    trainer._runtime_state = {"batches_seen": 6, "samples_seen": 48}
    output = object()
    monkeypatch.setattr(train.LtxvTrainer, "_training_step", lambda *_args: output)
    for _ in range(3):
        batch = {"video_latents": {"latents": torch.zeros(2, 1)}}
        assert trainer._training_step(batch) is output
    # Correct before the official loop's periodic checkpoint, not only after its callback.
    assert trainer._runtime_state == {"batches_seen": 9, "samples_seen": 72}


def test_production_bounded_stop_does_not_enable_diagnostics(monkeypatch, tmp_path):
    from ltx_outpaint import training_runtime

    def forbidden(*_args, **_kwargs):
        pytest.fail("Bounded stop must not imply diagnostics")

    monkeypatch.setattr(training_runtime, "run_training_with_health", forbidden)
    monkeypatch.setattr(training_runtime, "_memory", forbidden)
    trainer = _FakeTrainer(tmp_path, 5)
    checkpoint, _ = training_runtime.run_training(trainer, stop_after_step=6)
    assert checkpoint.name == "checkpoint-000006"
    assert trainer._global_step == 6
    assert trainer._runtime_state["samples_seen"] == 6
    assert not (tmp_path / "train_metrics.jsonl").exists()
    assert not list(tmp_path.glob("runtime_result_*.json"))
