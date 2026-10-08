"""CPU lossless-frame comparisons against the preserved strength_1p0 baseline."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "longrange_fhd_recurrence_test_20260923"
MODE = "strength_1p0_global_lineart_prior"
REGIONS = {"full": (0, 0, 1920, 1080), "left": (0, 0, 480, 1080), "right": (1440, 0, 1920, 1080)}


def frame(root, mode, index):
    chunk = 0 if index < 65 else 1 + (index - 65) // 48
    start = 0 if chunk == 0 else 65 + (chunk - 1) * 48
    path = root / "chunks" / mode / f"segment_{chunk:06d}.npy"
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return Image.fromarray(np.asarray(arr[index - start]))


def pair(before, after, region, index):
    box = REGIONS[region]
    width = 960 if region == "full" else 480
    a, b = before.crop(box), after.crop(box)
    height = round(a.height * width / a.width)
    a, b = a.resize((width, height)), b.resize((width, height))
    out = Image.new("RGB", (width * 2, height + 28), "#161616")
    out.paste(a, (0, 28)); out.paste(b, (width, 28))
    d = ImageDraw.Draw(out)
    d.text((8, 7), f"BASELINE 1.0 | F{index:06d} {index/24:.3f}s | {region}", fill="white")
    d.text((width + 8, 7), "LINEART GLOBAL | lambda .5/.3/.1", fill="#ffdc65")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--end-chunk", type=int, required=True)
    args = p.parse_args()
    if not 1 <= args.end_chunk <= 23:
        raise ValueError("end chunk must be C1..C23")
    last = 64 + args.end_chunk * 48
    phases = {}
    indices = {0}
    for c in range(1, args.end_chunk + 1):
        phases[c] = [48*c, 48*c+16, 48*c+17, 48*c+40, 48*c+64]
        indices.update(phases[c])
    timeline = [round(s*24) for s in (2, 4, 6, 8, 10, 20, 25, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48) if s*24 <= last]
    indices.update(timeline)
    outdir = ROOT / "comparisons" / f"C0_C{args.end_chunk}"
    sheetdir = ROOT / "contact_sheets" / f"C0_C{args.end_chunk}"
    sheetdir.mkdir(parents=True, exist_ok=True)
    for reg in REGIONS:
        (outdir / reg).mkdir(parents=True, exist_ok=True)
    for i in sorted(indices):
        before = frame(BASE, "strength_1p0", i)
        after = frame(ROOT, MODE, i)
        for reg in REGIONS:
            pair(before, after, reg, i).save(outdir / reg / f"F{i:06d}.jpg", quality=94)
    for reg in REGIONS:
        selections = {"timeline": timeline or [last]}
        selections.update({f"C{c:02d}_phases": vals for c, vals in phases.items()})
        for name, selection in selections.items():
            images = []
            for i in selection:
                im = Image.open(outdir / reg / f"F{i:06d}.jpg")
                im.thumbnail((960, 460))
                images.append(im.copy())
                im.close()
            sheet = Image.new("RGB", (max(x.width for x in images), sum(x.height for x in images)), "#161616")
            y = 0
            for im in images:
                sheet.paste(im, (0, y)); y += im.height
            sheet.save(sheetdir / f"{name}_{reg}.jpg", quality=91)
    report = {"baseline": str(BASE / "outputs/strength_1p0"), "experiment": str(ROOT / "outputs" / MODE),
              "end_chunk": args.end_chunk, "global_frame_indices": sorted(indices),
              "phase_offsets": {"start": 0, "prefix_last": 16, "new_first": 17, "new_middle": 40, "new_last": 64},
              "source": "lossless uint8 segment arrays, not MP4 decode", "comparisons": str(outdir), "contact_sheets": str(sheetdir)}
    (ROOT / "report" / f"comparison_C0_C{args.end_chunk}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
