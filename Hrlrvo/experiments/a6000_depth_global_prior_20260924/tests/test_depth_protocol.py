"""No GPU: prove the inherited sampler call and isolation/continuation contracts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def calls(path, suffix):
    tree = ast.parse(path.read_text())
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == suffix]


def test_sample_chunk_call_identical_to_h200_lineart():
    before = calls(ROOT/'verification/run_lineart_chain.reference.py', 'sample_chunk')
    after = calls(ROOT/'run_depth_chain.py', 'sample_chunk')
    assert len(before) == len(after) == 1
    assert ast.dump(before[0], include_attributes=False) == ast.dump(after[0], include_attributes=False)


def test_all_model_modules_are_the_exact_h200_snapshot():
    source = ROOT/'source/VideoOutpainting_test/main/src'
    expected = json.loads((ROOT/'verification/source_snapshot.json').read_text())['sha256']
    actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*.py')}
    assert actual == expected


def test_restore_after_model_load_before_generation():
    path = ROOT/'run_depth_chain.py'
    load = calls(path, '_load_models')[0]
    restore = calls(path, '_restore_post_chunk_rng')[0]
    generate = calls(path, 'sample_chunk')[0]
    assert load.lineno < restore.lineno < generate.lineno
    text = path.read_text()
    assert 'torch.equal(restored_cpu,resume.cpu_rng_state)' in text
    assert 'torch.equal(restored_cuda,resume.cuda_rng_state)' in text
    assert "if index<1:raise RuntimeError" in text
    assert "'c0_depth_enabled':False" in text


def test_pin_recurrence_and_resume_identity():
    text = (ROOT/'run_depth_chain.py').read_text()
    assert 'a.rebase_temporal_position and a.use_latent_carry and a.generated_prefix_strength == 1.0' in text
    assert 'config.seed!=71634910 or config.inference_steps!=30' in text
    assert 'a.depth_bias_scale!=0.5' in text
    for key in ['runner_sha256', 'source_snapshot_sha256', 'settings_sha256', 'c0_bootstrap_sha256']:
        assert f"identity_request['{key}']" in text
    assert 'reference_extraction_sha256' in (ROOT/'scripts/bootstrap_and_configure.py').read_text()


def test_launcher_has_only_gpu0_no_retry_or_foreign_signals():
    text = (ROOT/'start_depth.sh').read_text()
    assert 'CUDA_VISIBLE_DEVICES=0' in text
    assert 'flock -n 9' in text
    assert 'GPU_PROCESSES=$(nvidia-smi' in text
    assert 'grep -Fx "$uuid" <<< "$GPU_PROCESSES"' in text
    assert 'pkill' not in text and 'killall' not in text
    assert 'queue_arm.sh' not in text
    assert '--rebase-temporal-position --use-latent-carry --generated-prefix-strength 1.0' in text


def test_c0_recovery_is_fail_closed_not_rgb24_reconstruction():
    text = (ROOT/'scripts/bootstrap_and_configure.py').read_text()
    for value in ['c427021db8c691806f17d18fa4b2bd89919abd54dbaa3ae161af0c6e67ea2ec6',
                  '1eb955cb01755ae008574fa5a1fbf6b801deb2413b5c5d55799488568d1452e4',
                  '892a7f5d9a9b500300e6d994682ae670991019e95106d029b9e61d6ebf181d63']:
        assert value in text
    assert "replay.get(key) != digest" in text
    assert 'sha(replay_segment) != sha(segment_path)' in text
    assert 'sha(replay_carry) != record' in text
    assert 'refusing approximate continuation' in text


def test_no_stale_initial_outputs_or_silent_preview_failure():
    text = (ROOT/'run_depth_chain.py').read_text()
    assert 'Initial bootstrap requires empty new run directories' in text
    assert "previous_final.get('mp4_sha256')!=file_sha256(output)" in text
    assert 'STATE=preview_failed; CODE=$PREVIEW_CODE' in (ROOT/'start_depth.sh').read_text()
