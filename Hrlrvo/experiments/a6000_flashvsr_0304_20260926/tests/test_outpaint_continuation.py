import importlib.util
import fcntl
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1] / 'tools/outpaint_continuation.py'
SPEC = importlib.util.spec_from_file_location('continuation', FILE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class ContinuationTests(unittest.TestCase):
    def test_missing_process_is_finished(self):
        self.assertFalse(mod.process_matches(999999999, '1'))

    def test_live_process_requires_matching_start(self):
        import os
        ticks = Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19]
        self.assertTrue(mod.process_matches(os.getpid(), ticks))
        self.assertFalse(mod.process_matches(os.getpid(), 'wrong'))

    def test_busy_stage_lock_waits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / 'state/locks/outpaint.lock'
            path.parent.mkdir(parents=True)
            with path.open('w') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(mod.locks_free(root))
                fcntl.flock(handle, fcntl.LOCK_UN)
                self.assertTrue(mod.locks_free(root))

    def test_receipt_fails_closed_on_status_identity_and_content(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = {'input': {'video_id': '02'}, 'seed': 1, 'frames': {},
                      'preprocess': {}, 'outpainting': {},
                      'artifacts': {'outpaint': 'data/02'}}
            stage = root / 'data/02'
            output = stage / 'final/02_Outpaint_4K.mp4'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'existing-output')
            receipt_path = stage / 'metadata/stage_complete.json'
            receipt_path.parent.mkdir()
            receipt = {'status': 'PASS', 'request': {k: config[k] for k in
                       ['input', 'seed', 'frames', 'preprocess', 'outpainting']},
                       'output': str(output), 'frames': 1952, 'output_fps': 16,
                       'output_sha256': mod.sha256(output)}
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(mod.validate_completed_02(root, config)['status'], 'PASS')
            for key, value in [('status', 'FAIL'), ('frames', 100), ('output_sha256', 'bad')]:
                bad = dict(receipt, **{key: value})
                receipt_path.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):
                    mod.validate_completed_02(root, config)

    def test_no_signal_or_upscale_command(self):
        source = FILE.read_text()
        self.assertNotIn('os.kill(', source)
        self.assertNotIn('killpg(', source)
        self.assertIn("'--stage', 'outpaint', '--resume'", source)


if __name__ == '__main__':
    unittest.main()
