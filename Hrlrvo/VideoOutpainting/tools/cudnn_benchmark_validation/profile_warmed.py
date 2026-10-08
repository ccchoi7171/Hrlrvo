"""One cold + one warmed full-model step; Kineto only on the warmed F/B.

Avoid torch.profiler.events/key_averages on the full-model trace: its expensive
Python FunctionEvent tree is not needed for raw timestamp/kernel attribution.
"""

import contextlib
from pathlib import Path

import torch
import validate_training as validation

BaseInstrument = validation.Instrument


class WarmProfile(BaseInstrument):
    def __init__(self, directory, profile=False):
        self.selected = Path(directory).name == "step_02"
        super().__init__(directory, profile=self.selected)
        self.prof = None

    @contextlib.contextmanager
    def stage(self, name, sync=True):
        if self.selected and name == "forward":
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.prof.start()
        try:
            with super().stage(name, sync):
                yield
        finally:
            if self.selected and name == "backward_only" and self.prof is not None:
                self.prof.stop()

    def finish(self):
        super().finish()
        if self.prof is not None:
            print("EXPORT_RAW_WARM_TRACE", self.directory, flush=True)
            self.prof.export_chrome_trace(str(self.directory / "trace.json"))
            self.prof = None
            print("EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    validation.Instrument = WarmProfile
    validation.main()
