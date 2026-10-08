from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

import orchestrate


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


class CancellationSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        orchestrate._ACTIVE_CHILDREN.begin_run()
        self.temporary = tempfile.TemporaryDirectory(prefix="hrlrvo_cancel_test_")
        self.root = Path(self.temporary.name)
        self.logger = orchestrate.Logger(self.root / "orchestrator.log")

    def tearDown(self) -> None:
        orchestrate._ACTIVE_CHILDREN.terminate_all(grace_seconds=0.1)
        self.logger.close()
        self.temporary.cleanup()

    def _command_with_grandchild(self, pid_file: Path) -> list[str]:
        child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        parent_code = (
            "import os,subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
            f"open({str(pid_file)!r},'w').write(f'{{os.getpid()}} {{p.pid}}'); "
            "time.sleep(60)"
        )
        return [sys.executable, "-c", parent_code]

    def _wait_for_pids(self, path: Path) -> tuple[int, int]:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if path.is_file() and path.read_text(encoding="utf-8").strip():
                parent, grandchild = path.read_text(encoding="utf-8").split()
                return int(parent), int(grandchild)
            time.sleep(0.02)
        self.fail("dummy subprocess did not publish its PIDs")

    def test_registry_cancellation_kills_child_and_grandchild(self) -> None:
        pid_file = self.root / "pids.txt"
        result: list[int] = []
        worker = threading.Thread(
            target=lambda: result.append(
                orchestrate._run_logged(
                    "dummy",
                    self._command_with_grandchild(pid_file),
                    environment=os.environ.copy(),
                    model_log=self.root / "dummy.log",
                    logger=self.logger,
                )
            )
        )
        worker.start()
        parent, grandchild = self._wait_for_pids(pid_file)

        orchestrate._ACTIVE_CHILDREN.terminate_all(grace_seconds=0.2)
        worker.join(timeout=3.0)

        self.assertFalse(worker.is_alive())
        self.assertTrue(result and result[0] != 0)
        self.assertFalse(_running(parent))
        self.assertFalse(_running(grandchild))

    def test_concurrent_worker_exception_surfaces_and_cancels_sibling_group(self) -> None:
        pid_file = self.root / "sibling_pids.txt"

        def sibling() -> None:
            orchestrate._run_logged(
                "sibling",
                self._command_with_grandchild(pid_file),
                environment=os.environ.copy(),
                model_log=self.root / "sibling.log",
                logger=self.logger,
            )

        def failure() -> None:
            self._wait_for_pids(pid_file)
            raise RuntimeError("intentional worker failure")

        with self.assertRaisesRegex(RuntimeError, "intentional worker failure"):
            orchestrate._run_concurrently(
                [("sibling", sibling), ("failure", failure)], logger=self.logger
            )

        parent, grandchild = self._wait_for_pids(pid_file)
        self.assertFalse(_running(parent))
        self.assertFalse(_running(grandchild))

    def test_signal_while_main_waits_for_concurrent_worker_cancels_descendants(self) -> None:
        pid_file = self.root / "signal_pids.txt"

        def sibling() -> None:
            orchestrate._run_logged(
                "signal-sibling",
                self._command_with_grandchild(pid_file),
                environment=os.environ.copy(),
                model_log=self.root / "signal-sibling.log",
                logger=self.logger,
            )

        def send_signal() -> None:
            self._wait_for_pids(pid_file)
            os.kill(os.getpid(), signal.SIGINT)

        previous = signal.getsignal(signal.SIGINT)
        signal.signal(
            signal.SIGINT,
            lambda signum, _frame: (_ for _ in ()).throw(
                orchestrate._SignalCancellation(signum)
            ),
        )
        sender = threading.Thread(target=send_signal)
        sender.start()
        started = time.monotonic()
        try:
            with self.assertRaises(orchestrate._SignalCancellation):
                orchestrate._run_concurrently([("sibling", sibling)], logger=self.logger)
        finally:
            signal.signal(signal.SIGINT, previous)
            sender.join(timeout=3.0)

        parent, grandchild = self._wait_for_pids(pid_file)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertFalse(_running(parent))
        self.assertFalse(_running(grandchild))

    def test_signal_handler_ignores_duplicate_during_cleanup(self) -> None:
        previous = signal.getsignal(signal.SIGINT)
        handler = orchestrate._OneShotSignalHandler([signal.SIGINT])
        signal.signal(signal.SIGINT, handler)
        try:
            with self.assertRaises(orchestrate._SignalCancellation):
                handler(signal.SIGINT, None)
            # A forwarded duplicate must not interrupt TERM/KILL/reap cleanup.
            handler(signal.SIGINT, None)
            self.assertTrue(handler.triggered)
            self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
        finally:
            signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    unittest.main()
