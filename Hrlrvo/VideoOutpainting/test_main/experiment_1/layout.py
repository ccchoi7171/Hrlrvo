"""Pure timeline and artifact contracts for the immutable-bridge experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


OUTPUT_FRAMES = 1025
BRIDGE_FRAMES = 9
MIDDLE_FRAMES = 64
RIGHT_CONDITION_FRAMES = 8
INTERVAL_FRAMES = BRIDGE_FRAMES + MIDDLE_FRAMES + RIGHT_CONDITION_FRAMES
BRIDGE_STRIDE = BRIDGE_FRAMES + MIDDLE_FRAMES
BRIDGE_CONTEXT_FRAMES = 65
BRIDGE_EXTRACT_START = 25
BRIDGE_COUNT = 15
INTERVAL_COUNT = 14
TEMPORAL_FACTOR = 8


@dataclass(frozen=True)
class BridgePlan:
    index: int
    bridge_id: str
    global_start: int
    global_stop: int
    context_start: int
    context_stop: int
    extract_start: int
    extract_stop: int
    pad_left: int
    pad_right: int
    reuse_full_context: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BodyPlan:
    index: int
    body_id: str
    global_start: int
    global_stop: int
    left_bridge: str
    right_bridge: str
    middle_global_start: int
    middle_global_stop: int
    visible_middle_stop: int

    def to_dict(self) -> dict:
        return asdict(self)


def bridge_plans(*, reuse_b0: str | Path | None = None) -> tuple[BridgePlan, ...]:
    result = []
    for index in range(BRIDGE_COUNT):
        start = index * BRIDGE_STRIDE
        if index == 0:
            context_start, context_stop = 0, BRIDGE_CONTEXT_FRAMES
            extract_start = 0
        else:
            context_start = start - BRIDGE_EXTRACT_START
            context_stop = context_start + BRIDGE_CONTEXT_FRAMES
            extract_start = BRIDGE_EXTRACT_START
        pad_left = max(0, -context_start)
        pad_right = max(0, context_stop - OUTPUT_FRAMES)
        result.append(
            BridgePlan(
                index=index,
                bridge_id=f"B{index:02d}",
                global_start=start,
                global_stop=start + BRIDGE_FRAMES,
                context_start=context_start,
                context_stop=context_stop,
                extract_start=extract_start,
                extract_stop=extract_start + BRIDGE_FRAMES,
                pad_left=pad_left,
                pad_right=pad_right,
                reuse_full_context=str(Path(reuse_b0).resolve()) if index == 0 and reuse_b0 else None,
            )
        )
    plans = tuple(result)
    validate_bridge_plans(plans)
    return plans


def body_plans() -> tuple[BodyPlan, ...]:
    result = []
    for index in range(INTERVAL_COUNT):
        start = index * BRIDGE_STRIDE
        middle_start = start + BRIDGE_FRAMES
        middle_stop = middle_start + MIDDLE_FRAMES
        result.append(
            BodyPlan(
                index=index,
                body_id=f"I{index:02d}",
                global_start=start,
                global_stop=start + INTERVAL_FRAMES,
                left_bridge=f"B{index:02d}",
                right_bridge=f"B{index + 1:02d}",
                middle_global_start=middle_start,
                middle_global_stop=middle_stop,
                visible_middle_stop=min(middle_stop, OUTPUT_FRAMES),
            )
        )
    plans = tuple(result)
    validate_body_plans(plans)
    return plans


def validate_bridge_plans(plans: Iterable[BridgePlan]) -> None:
    rows = tuple(plans)
    if len(rows) != BRIDGE_COUNT:
        raise ValueError(f"Expected {BRIDGE_COUNT} bridges, got {len(rows)}")
    for expected, row in enumerate(rows):
        if row.index != expected or row.bridge_id != f"B{expected:02d}":
            raise ValueError("Bridge identities are not contiguous")
        if row.global_start != expected * BRIDGE_STRIDE or row.global_stop - row.global_start != BRIDGE_FRAMES:
            raise ValueError(f"{row.bridge_id} has an invalid global range")
        if row.context_stop - row.context_start != BRIDGE_CONTEXT_FRAMES:
            raise ValueError(f"{row.bridge_id} context is not 65 frames")
        if row.extract_stop - row.extract_start != BRIDGE_FRAMES:
            raise ValueError(f"{row.bridge_id} extraction is not 9 frames")
        if row.extract_start < 0 or row.extract_stop > BRIDGE_CONTEXT_FRAMES:
            raise ValueError(f"{row.bridge_id} extraction lies outside its context")
        real_start = max(0, row.context_start)
        real_stop = min(OUTPUT_FRAMES, row.context_stop)
        if real_stop - real_start + row.pad_left + row.pad_right != BRIDGE_CONTEXT_FRAMES:
            raise ValueError(f"{row.bridge_id} padding does not restore a 65-frame context")
    if rows[-1].pad_right != 37:
        raise ValueError("Terminal virtual bridge must record 37 repeated source frames")


def validate_body_plans(plans: Iterable[BodyPlan]) -> None:
    rows = tuple(plans)
    if len(rows) != INTERVAL_COUNT:
        raise ValueError(f"Expected {INTERVAL_COUNT} intervals, got {len(rows)}")
    for expected, row in enumerate(rows):
        if row.index != expected or row.body_id != f"I{expected:02d}":
            raise ValueError("Body identities are not contiguous")
        if row.global_stop - row.global_start != INTERVAL_FRAMES:
            raise ValueError(f"{row.body_id} is not an 81-frame operator")
        if row.middle_global_stop - row.middle_global_start != MIDDLE_FRAMES:
            raise ValueError(f"{row.body_id} middle is not 64 frames")
        if row.left_bridge != f"B{expected:02d}" or row.right_bridge != f"B{expected + 1:02d}":
            raise ValueError(f"{row.body_id} endpoints are not adjacent immutable bridges")


def assembly_occupancy() -> tuple[int, ...]:
    occupancy = [0] * OUTPUT_FRAMES
    for bridge in bridge_plans():
        for frame in range(max(0, bridge.global_start), min(OUTPUT_FRAMES, bridge.global_stop)):
            occupancy[frame] += 1
    for body in body_plans():
        for frame in range(body.middle_global_start, body.visible_middle_stop):
            occupancy[frame] += 1
    return tuple(occupancy)


def validate_contract() -> None:
    if INTERVAL_FRAMES != 81 or (INTERVAL_FRAMES - 1) % TEMPORAL_FACTOR:
        raise ValueError("The 9+64+8 operator must be an 81-frame VAE-aligned clip")
    if 1 + (INTERVAL_FRAMES - 1) // TEMPORAL_FACTOR != 11:
        raise ValueError("The body operator must contain exactly 11 latent frames")
    if 1 + (BRIDGE_FRAMES - 1) // TEMPORAL_FACTOR != 2:
        raise ValueError("The left bridge must occupy exactly two latent frames")
    if RIGHT_CONDITION_FRAMES // TEMPORAL_FACTOR != 1:
        raise ValueError("The right bridge condition must occupy exactly one latent frame")
    occupancy = assembly_occupancy()
    if len(occupancy) != OUTPUT_FRAMES or set(occupancy) != {1}:
        raise ValueError("Bridge/body assembly must cover every delivered frame exactly once")


validate_contract()

