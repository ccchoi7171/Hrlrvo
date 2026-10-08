"""Build an offline image gallery and native-area crop sheets; no inference."""
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference/candidates"


def labelled(image, title, width=960):
    h = round(image.height * width / image.width)
    tile = Image.new("RGB", (width, h + 30), "#151a22")
    tile.paste(image.resize((width, h), Image.Resampling.LANCZOS), (0, 30))
    ImageDraw.Draw(tile).text((9, 8), title, fill="white")
    return tile


def grid(tiles, columns, path):
    width, height = max(t.width for t in tiles), max(t.height for t in tiles)
    rows = (len(tiles) + columns - 1) // columns
    out = Image.new("RGB", (columns * width, rows * height), "#151a22")
    for i, tile in enumerate(tiles):
        out.paste(tile, ((i % columns) * width, (i // columns) * height))
    out.save(path)


def main():
    candidates = []
    for path in sorted((OUT / "depth").glob("D*/metadata.json")):
        meta = json.loads(path.read_text())
        directory = path.parent
        normal = np.load(directory / "prediction_display_normalized.npy", allow_pickle=False)
        near = normal if meta["prediction_kind"] == "relative" else 1 - normal
        # Pure display transformation to reveal low-contrast background values.
        # Raw prediction and ordinary linear preview are unchanged.
        enhanced = np.rint(np.sqrt(np.clip(near, 0, 1)) * 255).astype(np.uint8)
        cv2.imwrite(str(directory / "depth_structure_gamma05.png"), cv2.applyColorMap(enhanced, cv2.COLORMAP_TURBO))
        candidates.append((directory, meta))
    for display, suffix in [("depth_color.png", "color"), ("depth_gray.png", "gray"), ("depth_structure_gamma05.png", "structure_gamma05")]:
        tiles = [labelled(Image.open(d / display), m["name"] + " | WHITE/WARM = NEAR") for d, m in candidates]
        grid(tiles, 2, OUT / "depth" / f"contact_sheet_{suffix}.png")
    crops = [("left", (0, 320, 480, 850)), ("stage", (720, 320, 1200, 850)), ("right", (1440, 320, 1920, 850))]
    crop_tiles = []
    source = Image.open(ROOT / "reference/c0_frame000.png")
    for label, box in crops:
        crop_tiles.append(labelled(source.crop(box), "RGB / " + label, 480))
    for directory, meta in candidates:
        im = Image.open(directory / "depth_structure_gamma05.png")
        for label, box in crops:
            crop = im.crop(box)
            crop.save(directory / f"crop_{label}_structure.png")
            crop_tiles.append(labelled(crop, meta["name"] + " / " + label, 480))
    grid(crop_tiles, 3, OUT / "depth/contact_sheet_crops.png")

    parts = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>C0 Structure Prior Candidates</title><style>
body{font:16px system-ui,sans-serif;background:#111722;color:#e9edf3;margin:24px auto;max-width:1500px;padding:0 18px}h1,h2{color:#b8d7ff}a{color:#8ec5ff}p{line-height:1.7}nav{position:sticky;top:0;background:#111722;padding:12px;z-index:2;border-bottom:1px solid #445}nav a{margin-right:20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}.card{background:#1d2635;border:1px solid #3c4b62;border-radius:10px;padding:14px}.card img{width:100%;height:auto;background:#000}.links a{margin-right:12px;line-height:2}small{color:#b6c3d4}.source{max-width:960px;width:100%}code{color:#ffe39c}button{padding:5px 10px;margin:3px;background:#34435b;color:white;border:1px solid #71809a;cursor:pointer}.warning{border-left:4px solid #ffce73;padding:12px;background:#342e23}
</style><nav><a href="#lineart">LineArt 4종</a><a href="#depth">Depth 5종</a><a href="#sheets">비교 시트</a></nav>
<h1>C0 첫 프레임 — LineArt / Depth 후보</h1><p><b>추천: LineArt L2 detailed / 큰 구조용 Depth D5 V2 Base, 세부 대안 D4.</b> <a href="RECOMMENDATIONS.md">추천 근거와 한계</a></p>
<p>모든 후보는 동일한 1920×1080 C0 RGB 한 장에서 생성했습니다. <b>LongRange diffusion은 중지 상태</b>이며, 여기의 Depth 모델도 CPU에서만 실행했습니다. 이미지를 클릭하면 원해상도로 열립니다.</p>
<a href="../c0_frame000.png"><img class="source" src="../c0_frame000.png"></a>
<p class="warning">후보를 아직 active prior로 교체하거나 inference에 적용하지 않았습니다. 현재 Global의 9×15 <code>gᵢgⱼ</code> 방식은 선의 방향·연결성 대신 cell별 밀도만 보존합니다. 후보가 더 선명하다고 주입 효과가 보장되지는 않습니다.</p>
<h2 id="lineart">LineArt — native FHD, 얇은 Canny 선</h2><p>축소 후 확대 없이 FHD에서 직접 추출했습니다. dilation 없음. 확대 보기로 창틀·기둥의 선을 비교하세요.</p><div class="cards">"""]
    labels = {"balanced": "L1 · Balanced — 구조 중심, 보수적", "detailed": "L2 · Detailed — 상세도/노이즈 균형 추천", "very_detailed": "L3 · Very detailed — 텍스처까지 검출", "clahe_lowlight": "L4 · CLAHE lowlight — 어두운 부분 강화"}
    for name, title in labels.items():
        variants = [("검은 배경", f"lineart/{name}.png"), ("흰 배경", f"lineart/{name}_black_on_white.png"), ("RGB overlay", f"lineart/{name}_overlay.png"), ("Global 9×15", f"lineart/{name}_global_line_strength_heatmap.png")]
        buttons = "".join(f'<button onclick="document.getElementById(\'img_{name}\').src=\'{src}\';document.getElementById(\'link_{name}\').href=\'{src}\'">{label}</button>' for label, src in variants)
        links = "".join(f'<a href="{src}">{label}</a>' for label, src in variants)
        parts.append(f'<article class="card"><h3>{title}</h3>{buttons}<a id="link_{name}" href="lineart/{name}.png"><img id="img_{name}" src="lineart/{name}.png"></a><div class="links">{links}<a href="lineart/{name}.npy">float32 NPY</a></div></article>')
    parts.append('</div><p><a href="lineart/parameters_and_stats.json">LineArt 파라미터·통계</a> · <a href="lineart/visual_review.md">시각 검토</a></p><h2 id="depth">Depth — 4개 checkpoint / 5개 실제 추정 결과</h2><p>D1과 D4는 같은 V2 Small 모델의 입력 해상도 비교입니다. 흑백은 <b>흰색=가까움</b>, 검정=멀어짐이며, 컬러는 warm=가까움입니다. 모델마다 표시 범위를 별도로 정규화했습니다. Gamma 0.5는 배경의 작은 깊이 차이를 보기 위한 <b>시각화만</b> 바꾸며 새로운 추정 결과가 아닙니다. metric indoor 숫자도 이 무대에서 검증한 실제 거리는 아닙니다.</p><div class="cards">')
    for directory, meta in candidates:
        name = directory.name
        variants = [("Color", "depth_color.png"), ("Gray", "depth_gray.png"), ("Structure gamma0.5", "depth_structure_gamma05.png"), ("RGB overlay", "depth_overlay.png")]
        buttons = "".join(f'<button onclick="document.getElementById(\'img_{name}\').src=\'depth/{name}/{file}\';document.getElementById(\'link_{name}\').href=\'depth/{name}/{file}\'">{label}</button>' for label, file in variants)
        links = "".join(f'<a href="depth/{name}/{file}">{label}</a>' for label, file in variants)
        parts.append(f'<article class="card"><h3>{html.escape(name)}</h3><small>{html.escape(meta["model_id"])}<br>input {meta["actual_input_shape"]} · CPU {meta["inference_and_postprocess_seconds"]:.2f}s</small><br>{buttons}<a id="link_{name}" href="depth/{name}/depth_color.png"><img id="img_{name}" src="depth/{name}/depth_color.png"></a><div class="links">{links}<a href="depth/{name}/depth_raw.npy">Raw NPY</a><a href="depth/{name}/metadata.json">Metadata</a></div></article>')
    parts.append('''</div><h2 id="sheets">비교 시트 / Crop</h2><ul>
<li><a href="lineart/contact_sheet_fullview.png">LineArt 전체 + 기존 coarse</a></li>
<li><a href="lineart/contact_sheet_crops.png">LineArt 좌/우/무대/바닥 확대</a></li>
<li><a href="depth/contact_sheet_color.png">Depth 전체 컬러 비교</a></li>
<li><a href="depth/contact_sheet_gray.png">Depth 전체 흑백 비교</a></li>
<li><a href="depth/contact_sheet_structure_gamma05.png">Depth 배경 구조 contrast 비교</a></li>
<li><a href="depth/contact_sheet_crops.png">Depth 좌/우/무대 확대 + RGB</a></li></ul>
<h2>모델 출처</h2><p><a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf">Depth Anything V2 Small</a> · <a href="https://huggingface.co/LiheYoung/depth-anything-small-hf">Depth Anything V1 Small</a> · <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf">Depth Anything V2 Metric Indoor Small</a> · <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf">V2 Base (CC-BY-NC-4.0)</a></p><p>V2 Base weights are CC-BY-NC-4.0; this is a local research comparison, not a production/commercial deployment.</p></html>''')
    (OUT / "index.html").write_text("\n".join(parts))
    (OUT / "gallery_metadata.json").write_text(json.dumps({"depth_candidates": [m for _, m in candidates], "lineart_candidates": list(labels), "all_on_cpu": True, "diffusion_restarted": False}, indent=2) + "\n")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
