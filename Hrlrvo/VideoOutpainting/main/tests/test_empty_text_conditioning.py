from pathlib import Path
from types import SimpleNamespace

import pytest
from ltx_trainer.config import ValidationSample

from ltx_outpaint import inference
from ltx_outpaint.train import (
    OutpaintingTrainer,
    _validate_integration_config,
    load_training_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _inference_config(prompt="", negative_prompt=""):
    settings = inference._load_settings(ROOT / "configs/inference_4k.yaml")
    return inference._make_validation_config(
        settings, prompt, inference._vae_boundary(2160, 3840, 32, 32), 65,
        negative_prompt=negative_prompt,
    )


def test_resolved_train_inference_positive_and_negative_are_empty():
    config = _inference_config()
    train, outpainting = load_training_config(ROOT / "configs/train_4k.yaml")
    assert config.samples[0].prompt == outpainting["constant_prompt"] == ""
    assert config.negative_prompt == train.validation.negative_prompt == ""
    assert outpainting["conditions_dir"] is None
    assert config.generate_audio is False and train.training_strategy.audio is None
    assert config.video_cfg_scale == 3.0  # Keep native CFG; only text semantics change.


def test_inference_config_passes_positive_and_negative_text_to_native_config():
    prompt = "continuous extension of the same concert stage"
    negative = "backstage, crew, staff, audience"
    positive_only = _inference_config(prompt)
    assert positive_only.samples[0].prompt == prompt
    assert positive_only.negative_prompt == ""

    both = _inference_config(prompt, negative)
    assert both.samples[0].prompt == prompt
    assert both.negative_prompt == negative


@pytest.mark.parametrize("negative", [False, True])
def test_training_validation_rejects_nonempty_text(negative):
    config, outpainting = load_training_config(ROOT / "configs/train_4k.yaml")
    if negative:
        config.validation.negative_prompt = "official negative text must not be inserted"
    else:
        config.validation.samples = [ValidationSample(prompt="nonempty validation prompt")]
    with pytest.raises(ValueError, match="fixed-empty"):
        _validate_integration_config(config, outpainting)


@pytest.mark.parametrize(
    ("extra_args", "expected_prompt", "expected_negative"),
    [
        ([], "", ""),
        (["--prompt", "continuous extension of the same concert stage"],
         "continuous extension of the same concert stage", ""),
        (["--negative-prompt", "backstage, crew, staff, audience"],
         "", "backstage, crew, staff, audience"),
        (["--prompt", "continuous extension of the same concert stage",
          "--negative-prompt", "backstage, crew, staff, audience"],
         "continuous extension of the same concert stage", "backstage, crew, staff, audience"),
    ],
)
def test_cli_prompt_values_reach_inference_without_model_io(
    monkeypatch, extra_args, expected_prompt, expected_negative,
):
    monkeypatch.setattr("sys.argv", [
        "inference", "--config", "not-read.yaml", "--anchor-video", "not-read.mp4",
        "--output", "not-created.mp4", *extra_args,
    ])
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(inference, "sample_long_range", lambda *args, **kwargs: calls.append((args, kwargs)))
    inference.main()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[3] == expected_prompt
    assert kwargs["negative_prompt"] == expected_negative


def test_chunk_rejects_nonempty_negative_before_conditioning(monkeypatch):
    monkeypatch.setattr(inference, "_condition_latents", lambda *_: pytest.fail("Nonempty text reached conditioning"))
    runner = SimpleNamespace(_config=SimpleNamespace(samples=[SimpleNamespace(prompt="")], negative_prompt="bad"))
    with pytest.raises(ValueError, match="fixed-empty"):
        inference.sample_chunk(
            transformer=None, runner=runner, vae_encoder=None, anchor_frames=None,
            previous_prefix=None, boundary=None, anchor_rect=None, absolute_start_frame=0,
        )


def test_inference_yaml_negative_cannot_override_explicit_empty_cli_default():
    settings = inference._load_settings(ROOT / "configs/inference_4k.yaml")
    boundary = inference._vae_boundary(2160, 3840, 32, 32)
    settings["sampling"]["negative_prompt"] = "upstream negative"
    with pytest.raises(ValueError, match="--negative-prompt"):
        inference._make_validation_config(settings, "", boundary, 65)
    del settings["sampling"]["negative_prompt"]
    assert inference._make_validation_config(settings, "", boundary, 65).negative_prompt == ""


def test_inference_prompt_none_is_normalized_and_invalid_type_rejected():
    assert _inference_config(None, None).samples[0].prompt == ""
    assert _inference_config(None, None).negative_prompt == ""
    with pytest.raises(TypeError, match="prompt"):
        _inference_config(123)


@pytest.mark.parametrize("negative", [False, True])
def test_trainer_constructor_rejects_nonempty_text_before_model_loading(negative):
    bad = SimpleNamespace(
        samples=[SimpleNamespace(prompt="" if negative else "bad")],
        negative_prompt="bad" if negative else "",
    )
    with pytest.raises(ValueError, match="fixed-empty"):
        OutpaintingTrainer(SimpleNamespace(validation=bad), {})


def test_runner_accepts_inference_text_without_loading_models(monkeypatch):
    from ltx_trainer.validation_runner import ValidationRunner

    captured = []
    monkeypatch.setattr(ValidationRunner, "__init__", lambda self, **kwargs: captured.append(kwargs["config"]))
    config = _inference_config("stage extension", "backstage")
    inference.GeometryValidationRunner.create(config, Path("not-read"), Path("not-read"), None)
    assert captured == [config]
