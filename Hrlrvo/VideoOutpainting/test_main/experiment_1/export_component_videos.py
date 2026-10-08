"""CPU export of completed body tensors and reuse of existing bridge videos."""

from __future__ import annotations

import html
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from export_bridge_rgb import ROOT, encode_mp4, probe_video, sha256_file, tensor_sha256


def verified_video(path: Path, frames: int) -> dict:
    probe = probe_video(path)
    expected = {"width": 1920, "height": 1080, "fps": "24/1", "frames": frames}
    if probe != expected:
        raise RuntimeError(f"Invalid video {path}: {probe}")
    return {"path": str(path), "sha256": sha256_file(path), **probe}


def export_body(row: dict) -> dict:
    directory = ROOT / "bodies" / row["body_id"]
    status = json.loads((directory / "status.json").read_text())
    source = directory / "generated_81f_fp32.npy"
    frames = np.load(source, mmap_mode="r")
    if frames.shape != (81, 3, 1080, 1920) or frames.dtype != np.float32:
        raise RuntimeError(f"Invalid body tensor: {source}")
    raw_sha = tensor_sha256(frames)
    if status.get("status") != "PASS" or status.get("output_sha256") != raw_sha:
        raise RuntimeError(f"Body is not a hash-verified PASS artifact: {source}")
    destination = directory / "generated_81f.mp4"
    if not destination.exists():
        temporary = directory / "generated_81f.partial.mp4"
        encode_mp4(temporary, frames)
        verified_video(temporary, 81)
        temporary.replace(destination)
    record = {"body": row, "source": str(source), "source_tensor_sha256": raw_sha,
              "video": verified_video(destination, 81)}
    print(json.dumps({"status": "PASS", "body": row["body_id"], "video": str(destination)}), flush=True)
    return record


def link_bridge(record: dict) -> dict:
    directory = ROOT / "bridge_bank" / record["bridge_id"]
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key, count in (("context_65f_mp4", 65), ("bridge_9f_mp4", 9)):
        source = Path(record["outputs"][key])
        verified = verified_video(source, count)
        destination = directory / source.name
        if destination.exists():
            if not os.path.samefile(source, destination) and sha256_file(destination) != verified["sha256"]:
                raise RuntimeError(f"Existing bridge video differs: {destination}")
        else:
            os.link(source, destination)
        outputs[key] = {**verified, "path": str(destination), "reused_from": str(source)}
    print(json.dumps({"status": "PASS", "bridge": record["bridge_id"]}), flush=True)
    return {"bridge_id": record["bridge_id"], "global_bridge_range": record["global_bridge_range"],
            "context_range": [record["context_start"], record["context_stop"]],
            "padding": record["padding"], "videos": outputs}


def gallery(bodies: list[dict], bridges: list[dict]) -> str:
    cards = []
    for record in bodies:
        row = record["body"]
        cards.append((row["body_id"], f"global [{row['global_start']}, {row['global_stop']}) · 9f + 64f + 8f",
                      record["video"]["path"]))
    body_cards = len(cards)
    for record in bridges:
        for key, label in (("context_65f_mp4", "65f context"), ("bridge_9f_mp4", "extracted 9f bridge")):
            cards.append((record["bridge_id"] + " · " + label,
                          f"bridge global {record['global_bridge_range']} · padding {record['padding']}",
                          record["videos"][key]["path"]))
    rendered = []
    for index, (title, detail, path) in enumerate(cards):
        if index == body_cards:
            rendered.append('</div><h2>Bridge bank · 15 bridges</h2><div class="grid">')
        relative = html.escape(str(Path(path).relative_to(ROOT)), quote=True)
        rendered.append(f'<article><h3>{html.escape(title)}</h3><p>{html.escape(detail)}</p>'
                        f'<video controls preload="none" src="{relative}"></video>'
                        f'<p><a href="{relative}">Open MP4</a></p></article>')
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><title>Experiment 1 component videos</title>' \
        '<style>body{font:16px sans-serif;margin:24px;background:#15171a;color:#eee}a{color:#9ecbff}' \
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}' \
        'article{background:#24272c;padding:16px;border-radius:10px}video{width:100%}p{color:#bbb}' \
        '@media(max-width:480px){.grid{grid-template-columns:1fr}}</style>' \
        '<h1>Experiment 1 · component videos</h1><p>1920×1080 · 24fps · CPU export of saved results. ' \
        'Bodies: 81f / 3.375s. Bridge contexts: 65f / 2.708s. Extracted bridges: 9f / 0.375s. ' \
        'Intervals use an exclusive end. I13 and B14 include repeated last-frame padding beyond frame 1024.</p>' \
        '<h2>Bodies · 14 videos</h2><div class="grid">' + '\n'.join(rendered) + '</div></html>\n'


def main() -> None:
    plan = json.loads((ROOT / "plan.json").read_text())
    existing = json.loads((ROOT / "bridge_rgb" / "manifest.json").read_text())
    if len(plan["bodies"]) != 14 or existing.get("status") != "PASS" or len(existing["bridges"]) != 15:
        raise RuntimeError("Incomplete body plan or bridge export")
    bodies, bridges = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(export_body, row) for row in plan["bodies"]]
        for future in as_completed(futures):
            bodies.append(future.result())
        futures = [executor.submit(link_bridge, row) for row in existing["bridges"]]
        for future in as_completed(futures):
            bridges.append(future.result())
    bodies.sort(key=lambda item: item["body"]["index"])
    bridges.sort(key=lambda item: item["bridge_id"])
    index = ROOT / "component_videos.html"
    index.write_text(gallery(bodies, bridges), encoding="utf-8")
    manifest = {"status": "PASS", "body_count": len(bodies), "bridge_count": len(bridges),
                "video_count": len(bodies) + 2 * len(bridges), "cpu_workers": 4,
                "gallery": str(index), "bodies": bodies, "bridges": bridges}
    (ROOT / "component_videos_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("bodies", "bridges")}), flush=True)


if __name__ == "__main__":
    main()
