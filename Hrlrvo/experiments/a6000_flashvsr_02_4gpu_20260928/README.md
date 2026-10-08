# A6000 Video 02 Four-GPU FlashVSR

Isolated deployment overlay for video 02. It reuses the byte-pinned package, model assets,
and environment from the completed 03/04 A6000 experiment read-only; all run/status/output
paths are private to this root. Global tiles remain 0..169 with tile256/overlap32 and each
shard processes the full 1952-frame causal stream.
