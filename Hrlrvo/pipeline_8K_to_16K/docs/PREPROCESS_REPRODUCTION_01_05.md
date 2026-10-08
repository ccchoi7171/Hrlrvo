# Preprocess Reproduction 01–05

## VERDICT

**PREPROCESS REPRODUCED — RUNTIME-VERIFIED (CPU / FFmpeg only).**

- 01~05 각각 **first65 RGB24 = 65/65 exact**, **full RGB24 = 1952/1952 exact**.
- 모든 metadata 비교 PASS. 추가로 **MP4 container SHA256도 5/5 동일**하다. Container equality는 필수 gate를 대체하지 않으며, 전체 decoded RGB hash 비교를 별도로 수행했다.
- 모든 ID는 실제 공통 `run_preprocess_stage()`로 현재 8K source를 재처리했다. Canonical PreProc을 결과 대신 복사해서 통과시킨 검사가 아니다. 실제 input argv, FFmpeg 로그, producer identity, full decode 결과를 보존했다.
- Canonical 8K / PreProc MP4 및 원래 sidecar는 보존되었다. 생성물은 `data/validation/preprocess_reproduction/<ID>/`에만 있다.
- GPU / CUDA model load / Outpainting / FlashVSR / 실제 batch queue: **실행하지 않음**. 이 결과는 전처리 재현성의 PASS이며, 05 Outpainting seed 재현성의 PASS를 뜻하지 않는다.

### Historical contract

```text
Historical source: ../pipeline/src/00to01/preproc_8k_to_2k_24.py
Historical CRF: 18
Previous certification draft: CRF1
Corrected historical reproduction contract: CRF18
Reason: historical PreProc reproduction requirement
CRF only change: NO
```

이전 CRF1은 앞선 certification 요구를 구현한 값이었지만 historical seed-search 입력 재현 요구와 달랐다. 사용자가 그 요구를 명시적으로 정정했다. 과거 05 CRF1 결과의 0/65 불일치는 삭제하거나 새 결과로 바꾸지 않았다. `docs/REPRODUCIBILITY_05.md`의 archived 기록 및 `state/preprocess_reproduction/before/`를 유지한다.

## 1. Source of truth와 수정 범위

READ ONLY 기준:

- `/home/work/ai-graphics_33/Hrlrvo/pipeline/src/00to01/preproc_8k_to_2k_24.py`: `VIDEO_CRF` L35, `build_ffmpeg_command()` L124–179, `preprocess()` L181+.
- `/home/work/ai-graphics_33/Hrlrvo/pipeline/config/01/01_preproc_2K.sh`부터 `config/05/05_preproc_2K.sh`까지 실제 historical command 기록.
- 복사된 canonical `<ID>_PreProc_2K.json`: source/output SHA256에 연결된 `ffmpeg_command`, `crf`, 검증 metadata.

전체 builder argv를 5개 historical sidecar와 비교하는 CPU 회귀 테스트를 먼저 작성했다. 수정 전 불일치 로그는 `state/preprocess_reproduction/regression_before.log`, 수정 후 검증은 `cpu_tests.log`에 남겼다. 입력/출력 경로만 새 workspace/validation 경로로 바꾸었으며 인코딩 argv는 historical과 일치한다.

수정한 계산 구현은 **`src/cert_pipeline/preprocess.py` 한 곳**이다. `run_cert.py`는 이 contract의 validation, resume/provenance, 생성물 선택/발행만 수정했다. 모든 Full/Quick JSON과 job schema의 preprocess contract를 함께 바꿨으며, 기존 11개 config의 나머지 필드는 변경하지 않았다. 검증용 5개 JSON은 동일 model/seed/frame 조건에서 출력 경로만 격리했다.

## 2. 전체 preprocessing semantics 비교

| 항목 | 이전 certification draft | Historical / corrected | 판단 |
|---|---|---|---|
| input mapping | video0; audio 존재 시 audio0 | `-map 0:v:0 -map 0:a:0`, 정확히 1 audio 요구 | audio 없는 입력의 허용 여부 변경; 실제 01~05는 모두 audio 존재 |
| scale | `1920:1080:flags=lanczos:in_range=pc:out_range=pc` | 동일 | 유지 |
| SAR | `setsar=1` | 동일 | 유지 |
| retime / FPS | `setpts=PTS*2/3,fps=24` | 동일 | 유지 |
| fps_mode / vsync | `-fps_mode:v passthrough` / 추가 vsync 없음 | 동일 | 유지 |
| encoder | libx265 | libx265 | 유지 |
| preset | fast | fast | 유지 |
| CRF | **1** | **18** | 변경 |
| pix_fmt | yuv420p10le | yuv420p10le | 유지 |
| decoder threads | 미지정 | **`-threads 8` before `-i`** | 변경 |
| x265 params / pools | `log-level=error:pools=8` | 동일 | 유지; 별도 tile/model 변경 없음 |
| color_range | pc | pc | 유지 |
| colorspace / primaries | bt709 / bt709 | 동일 | 유지 |
| color_transfer | override 없음 | override 없음 (`null` contract) | 유지; 임의 `-color_trc` 추가하지 않음 |
| video codec tag | hvc1 | hvc1 | 유지 |
| track timescale | 24576 | 24576 | 유지 |
| metadata mapping | `-map_metadata 0` | 동일 | 유지 |
| audio speed / codec / bitrate | `atempo=1.5`, AAC, 192k | 동일 | 유지 |
| trim / frame cap | 없음, source 전체 처리 | 동일 | Full/Quick 모두 Stage0001은 전체 source |
| FFmpeg loglevel | info | **error** | historical argv 일치; stage 자체 로그는 보존 |
| strict input decode flags | `-xerror -err_detect explode` | 동일 | 유지 |
| output collision | `-y` | **`-n`** | historical partial-file 보호 |
| filename | `_Input_FHD_24fps.mp4`, PID temp | **`_PreProc_2K.mp4`, `.partial.mp4`** | historical naming |
| input count/validation | 반복 source `-count_frames`, 느슨한 color/audio 검사 | header `nb_frames` 필수; pc/bt709/10-bit/1 audio 검사 | historical preflight 복원 |
| output verification | 기존 metadata/count 검사 | metadata + strict full video/audio decode, N 확인 | historical 검증 경계 복원 |
| resume provenance | low-level source identity 중심 | source + 전체 contract + 구현 SHA + tool versions | CRF1 결과 재사용 차단 |
| canonical publication | canonical 이름으로 새 결과 발행 가능 | imported canonical 보존, 별도 generated path 선택 | reference 불변 요구 충족 |

CRF 이외의 차이를 맞췄으며, 이미 historical과 같았던 옵션을 불필요하게 바꾸지 않았다. source 해석과 output 검증은 현재 01~05 historical 입력의 계약을 따른다. 무음/다른 colorspace/임의 frame-rate 입력은 이 계약의 대상이 아니며 fail-closed 한다.

### 실제 producer command template

```bash
/usr/bin/ffmpeg -hide_banner -nostdin -loglevel error \
  -xerror -err_detect explode -threads 8 -i "$INPUT_8K" \
  -map 0:v:0 -map 0:a:0 \
  -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' \
  -fps_mode:v passthrough -c:v libx265 -preset fast -crf 18 \
  -pix_fmt yuv420p10le -x265-params 'log-level=error:pools=8' \
  -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 \
  -af atempo=1.5 -c:a aac -b:a 192k -map_metadata 0 \
  -video_track_timescale 24576 -n "$WORK_OUTPUT/<ID>_PreProc_2K.partial.mp4"
```

`fps=24` filter를 제거하거나 다른 timing algorithm으로 교체하지 않았다. 이상적 CFR 16 PTS를 2/3로 줄여 CFR24로 만드는 historical 순서 그대로다. 이번 다섯 영상에서는 N=1952 유지와 canonical 전체 decoded payload 일치가 **실제 검증**되었다. 이것이 임의 다른 타임스탬프 입력까지 증명하는 것은 아니다.

## 3. 하나의 구현 / canonical 보존

```text
run_cert.sh -> run_cert.py:main() [L1117+]
  -> validate_config() [L111+]
  -> execute_stages() [L1101+]
       all (Full or Quick) -> run_preprocess_stage() [L918–989]
       stage=preprocess   -> run_preprocess_stage() [same]
                              -> preflight_source() [L275+]
                              -> preprocess.run_preprocess() [L522+]
                                   -> build_ffmpeg_command() [L302–376]
                                   -> CPU FFmpeg historical encode
                                   -> validate_output_probe() [L243+]
                                   -> verify_full_decode() [L422–459]
                              -> SHA-checked publication + producer receipt
       stage=outpaint -> validate_preprocessed() [L809+]
                              -> preprocessed_input_path() [L690+]
```

- Fresh Full/Quick는 8K에서 다시 시작한다. canonical이 존재한다는 이유로 Stage0001을 skip하지 않는다.
- Canonical reference: `data/01_PreProc_2K_1920x1080/<ID>_PreProc_2K.mp4`와 원래 JSON. 이번 실행 및 이후 fresh Full에서도 덮어쓰지 않는다.
- Normal mutable stage output: `data/01_PreProc_2K_1920x1080/metadata/<ID>/generated/<ID>_PreProc_2K.mp4`와 receipt. Config의 선언된 canonical/cache path는 유지하고 `preprocess_output_path()`(L678+)가 publication path를 산출한다.
- Full의 다음 Outpaint와 stage-only Outpaint는 모두 같은 `preprocessed_input_path()`를 거쳐 generated artifact를 우선 사용한다. Generated artifact가 없을 때만 canonical cache를 선택한다. 깨진/부적합 generated receipt는 실패시키며 몰래 canonical로 fallback하지 않는다.
- 실제 선택은 `resolved_config.json.resolved_preprocess_artifact`에 기록한다. 이전 generated artifact를 갱신하면 metadata/preserved에 보존한다.
- 이번 5개 validation config는 처음부터 `data/validation/preprocess_reproduction/<ID>/`에 별도 발행한다.
- `--resume`는 source SHA/stat + 전체 corrected contract + `preprocess.py` SHA + FFmpeg/FFprobe version identity 및 artifact provenance를 검증한다. 과거 CRF1 receipt는 cache hit가 아니다.
- Batch queue는 **변경하지 않았다**. 기존 stage-only Outpaint를 통해 동일 PreProc selector/validator를 사용한다. Batch에는 별도의 preprocessing 구현/자동 생성이 없다. 준비가 필요하면 동일 `--stage preprocess`를 먼저 사용한다.

## 4. 실제 검증 절차와 환경

`tools/validate_preprocess_reproduction.py`는 생산 알고리즘이 아니라 CPU 검증 harness다. 실제 `run_cert.sh --stage preprocess` child process를 호출하고, 그 결과와 canonical을 독립적으로 decode/hash한다. 실제 실행은 ID 순서대로 한 번씩 수행했다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR="$PWD/state/preprocess_reproduction/tmp" \
/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python -B \
  tools/validate_preprocess_reproduction.py
```

각 child는 다음과 같은 **실제 entrypoint**를 사용했다.

```bash
./run_cert.sh --config configs/validation/preprocess_reproduction/<ID>.json --stage preprocess
```

Harness에서 `CUDA_VISIBLE_DEVICES=""`, 기존 Hrlrvo Python, `PYTHONDONTWRITEBYTECODE=1`을 child에 전달했다. Environment를 만들거나 package를 설치/변경하지 않았다. CPU encoder는 `/usr/bin/ffmpeg` **6.1.1-3ubuntu5**, probe도 동일 배포 version이다. 전체 version 문자열/빌드 옵션은 각 `producer.json`에 저장되어 있다.

검증 순서:

1. 현재 source와 canonical SHA를 historical sidecar의 source/output SHA와 대조한다.
2. 별도 validation destination이 없음을 확인한 후 실제 공통 Stage0001을 실행한다. 기존 검증 결과를 자동 overwrite하지 않는다.
3. Stage 자체에서 source header 검사, full CPU encode, output metadata/duration 검사, 전체 video+audio decode와 frame count를 검증한다.
4. Canonical/재생성 metadata(resolution, fps, frame count, codec, pix_fmt, time_base)를 비교한다.
5. 양쪽 **전체 video**를 동일 `rgb24` format으로 decode하여 순서별 payload size/hash를 비교한다. 처음 65개 및 전체를 각각 gate로 평가한다.
6. input/canonical SHA를 실행 후 다시 확인한다. 생성 결과의 SHA와 config SHA도 기록한다.

### Decoded RGB 비교 command

```bash
ffmpeg -hide_banner -nostdin -v error -xerror -err_detect explode \
  -threads 8 -i "$VIDEO" -map 0:v:0 -an -fps_mode passthrough \
  -pix_fmt rgb24 -f framemd5 -
```

서로 같은 CPU decoder/conversion argv를 사용한다. `framemd5`의 순서별 payload byte size와 hash를 비교했고, container metadata나 bitrate 유사성을 exact 판정에 쓰지 않았다. 양쪽 decoded row 수가 예상 N과 같은지도 요구한다. 오디오 sample equality가 별도 필수 gate는 아니지만, 전체 audio decode PASS와 모든 MP4의 byte identity는 추가로 관찰되었다.

## 5. Acceptance result — 01~05

**Source count는 각 파일 metadata와 decode로 확인한 값이다. Stage 구현에 1952를 하드코딩하지 않았다.**

| ID | Metadata | First65 RGB24 | Full RGB24 | First mismatch | MP4 SHA equal | Result (UTC) |
|---|---|---|---|---|---|---|
| 01 | PASS | 65/65 exact | 1952/1952 exact | none | YES | 2026-09-23T22:02:15.884055+00:00 |
| 02 | PASS | 65/65 exact | 1952/1952 exact | none | YES | 2026-09-23T22:04:37.016300+00:00 |
| 03 | PASS | 65/65 exact | 1952/1952 exact | none | YES | 2026-09-23T22:06:59.110671+00:00 |
| 04 | PASS | 65/65 exact | 1952/1952 exact | none | YES | 2026-09-23T22:09:37.821466+00:00 |
| 05 | PASS | 65/65 exact | 1952/1952 exact | none | YES | 2026-09-23T22:12:03.321293+00:00 |

모든 5개: `1920×1080`, `24/1 fps`, `1952 frames`, `HEVC/hvc1`, `yuv420p10le`, `time_base=1/24576`, full-range pc / bt709. 영상별 상세는 아래 comparison JSON과 source/output ffprobe JSON에 기록했다.

### 01

- Source: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/00_Input_8K_7680x4320/01_Input_8K.mp4`
- Source SHA256: `2d73992d09e37442fa7f0821711be0bb69584c6c5683ead99a8ad72ffe218d0c`
- Canonical: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/01_PreProc_2K.mp4`
- Canonical SHA256: `228d3e31e72805798a3ba22a4d0cac47b79f7c1d1d41c7f3ede35498f770a85a`
- Regenerated: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/validation/preprocess_reproduction/01/01_PreProc_2K.mp4`
- Regenerated SHA256: `228d3e31e72805798a3ba22a4d0cac47b79f7c1d1d41c7f3ede35498f770a85a`
- Config: `configs/validation/preprocess_reproduction/01.json`
- Config SHA256: `4642a44ad0de9e8ac72bcff6173e283bfe8ef308a47f96f8e56a64ca3716cf5e`
- First65 decoded: **65/65 exact**
- Full decoded: **1952/1952 exact**
- Source/canonical unchanged: **PASS**
- Evidence: [`comparison.json`](../data/validation/preprocess_reproduction/01/comparison.json), [`stage_terminal.log`](../data/validation/preprocess_reproduction/01/stage_terminal.log).

### 02

- Source: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/00_Input_8K_7680x4320/02_Input_8K.mp4`
- Source SHA256: `c192abcb6426be964a9a258c0c06fb22a81317eb8771d46fd81085a2172981b1`
- Canonical: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/02_PreProc_2K.mp4`
- Canonical SHA256: `58fa85818926402d6827fc59f52713867794ce6e4ed6e6805a05d7f171eb9ccb`
- Regenerated: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/validation/preprocess_reproduction/02/02_PreProc_2K.mp4`
- Regenerated SHA256: `58fa85818926402d6827fc59f52713867794ce6e4ed6e6805a05d7f171eb9ccb`
- Config: `configs/validation/preprocess_reproduction/02.json`
- Config SHA256: `36b3f5e1b14d01b3db5c6fa19ef31aaeef8dcd1c8c69fb8bcc751e4d36f10bf7`
- First65 decoded: **65/65 exact**
- Full decoded: **1952/1952 exact**
- Source/canonical unchanged: **PASS**
- Evidence: [`comparison.json`](../data/validation/preprocess_reproduction/02/comparison.json), [`stage_terminal.log`](../data/validation/preprocess_reproduction/02/stage_terminal.log).

### 03

- Source: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/00_Input_8K_7680x4320/03_Input_8K.mp4`
- Source SHA256: `d4326b28039cfdc51defca0759138dd38db843f7dfd59ed41c1846d4a73e7bbf`
- Canonical: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/03_PreProc_2K.mp4`
- Canonical SHA256: `7b2b12ffb5eb8a9854fc121321b15d8b464ee976ed7830b9762282e01601a296`
- Regenerated: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/validation/preprocess_reproduction/03/03_PreProc_2K.mp4`
- Regenerated SHA256: `7b2b12ffb5eb8a9854fc121321b15d8b464ee976ed7830b9762282e01601a296`
- Config: `configs/validation/preprocess_reproduction/03.json`
- Config SHA256: `c503fa5d8df9b5abb4294736f2bb49d183365ee6e3a880eb361608128325ffc5`
- First65 decoded: **65/65 exact**
- Full decoded: **1952/1952 exact**
- Source/canonical unchanged: **PASS**
- Evidence: [`comparison.json`](../data/validation/preprocess_reproduction/03/comparison.json), [`stage_terminal.log`](../data/validation/preprocess_reproduction/03/stage_terminal.log).

### 04

- Source: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/00_Input_8K_7680x4320/04_Input_8K.mp4`
- Source SHA256: `9341bf821851f87845e0b0d16426d05035574e688536ca587fa4e98fc0a338a1`
- Canonical: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/04_PreProc_2K.mp4`
- Canonical SHA256: `b671e890942a4f2e8e869f383cb73671fd6549ad79bcc92c965d87dcfbd8f393`
- Regenerated: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/validation/preprocess_reproduction/04/04_PreProc_2K.mp4`
- Regenerated SHA256: `b671e890942a4f2e8e869f383cb73671fd6549ad79bcc92c965d87dcfbd8f393`
- Config: `configs/validation/preprocess_reproduction/04.json`
- Config SHA256: `26c5c3e6d9045780a8ee3f3d6a4468a71b176607c838db6336995565fcb31342`
- First65 decoded: **65/65 exact**
- Full decoded: **1952/1952 exact**
- Source/canonical unchanged: **PASS**
- Evidence: [`comparison.json`](../data/validation/preprocess_reproduction/04/comparison.json), [`stage_terminal.log`](../data/validation/preprocess_reproduction/04/stage_terminal.log).

### 05

- Source: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/00_Input_8K_7680x4320/05_Input_8K.mp4`
- Source SHA256: `684b400a0057478295e22ddf7aa5caf5b5ad1a4fb77fafdbf9a5c74baa2cb454`
- Canonical: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4`
- Canonical SHA256: `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400`
- Regenerated: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/validation/preprocess_reproduction/05/05_PreProc_2K.mp4`
- Regenerated SHA256: `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400`
- Config: `configs/validation/preprocess_reproduction/05.json`
- Config SHA256: `995fb70c067e513120ba730361a89184a3bad4cabbdb3cfccba26ba55d873a45`
- First65 decoded: **65/65 exact**
- Full decoded: **1952/1952 exact**
- Source/canonical unchanged: **PASS**
- Evidence: [`comparison.json`](../data/validation/preprocess_reproduction/05/comparison.json), [`stage_terminal.log`](../data/validation/preprocess_reproduction/05/stage_terminal.log).

### 03 source version 경계

현재 pair는 `03_Input_8K.mp4` SHA **`d4326b28039cfdc51defca0759138dd38db843f7dfd59ed41c1846d4a73e7bbf`** ↔ `03_PreProc_2K.mp4` SHA **`7b2b12ffb5eb8a9854fc121321b15d8b464ee976ed7830b9762282e01601a296`**이다. 초기 seed-search의 pre-crop backup source를 사용하지 않았다. 위 현재 source SHA가 현재 canonical sidecar의 `input_sha256`와 일치함을 실제 실행 전에 확인했고, 현재 8K에서 다시 생성한 전체 RGB가 canonical과 일치했다.

## 6. Artifact tree와 저장 시점

```text
data/validation/preprocess_reproduction/
├── summary.json                         # 5개 최종 PASS; 영상별 전체 evidence
└── <ID>/
    ├── <ID>_PreProc_2K.mp4               # actual stage publication (canonical 아님)
    ├── <ID>_PreProc_2K.json              # source/producer/output SHA-bound receipt
    ├── stage_terminal.log                # real run_cert child stdout/stderr
    ├── canonical_full_rgb24.framemd5      # canonical 전체 RGB decode
    ├── regenerated_full_rgb24.framemd5    # generated 전체 RGB decode
    ├── comparison.json                   # metadata/hash/count gate 결과
    └── metadata/
        ├── logs/preprocess.log           # actual FFmpeg command + stage logs
        └── runs/<UTC stamp>/
            ├── output/<ID>_PreProc_2K.mp4 # partial encode -> verify -> rename
            └── metadata/
                ├── input_ffprobe.json
                ├── output_ffprobe.json
                ├── producer.json         # complete argv / contract / versions
                ├── full_decode.json      # strict video+audio decode, N PASS
                └── frame_alignment.json  # source/producer/count/timing receipt
```

`state/runs/<ID>_preprocess_reproduction/`에는 requested/resolved config가 있고, run-level log는 `logs/<ID>_preprocess_reproduction.log`다. Low-level work output과 발행 copy를 모두 보존했다. 실패했던 과거 `data/validation/05_preprocess_from_8k/` 기록은 이번 결과와 별개이며 변경하지 않았다.

## 7. Verification / changed scope

- Python AST/syntax: **PASS**.
- Ruff (변경 Python + 관련 CPU tests, `--no-cache`): **PASS**.
- Shell syntax (`run_cert.sh`, `run_batch.sh`): **PASS**.
- JSON/schema + 실제 config validator: **16/16 PASS**.
- CPU unit tests: **32 PASS**. Shared dispatch, complete historical argv, input/output contract, CRF1 resume rejection, producer toolchain drift rejection, canonical immutability를 포함한다. Batch test는 mock child/임시 디렉터리의 상태 머신 검증이며 실제 queue 실행이 아니다.
- 5개 actual output의 producer implementation SHA와 현재 preprocess.py SHA 일치: **PASS**.
- 기존 11개 config의 preprocess 이외 필드: **UNCHANGED**. Seeds, model paths, prompts, 30 steps, frames, feather/final settings는 유지했다.
- `run_outpaint_stage`, `run_upscale_stage`, `_run_flash_native`, `seed_all`, `upscale_input`: 수정 전과 **AST-identical**.
- `src/cert_pipeline/outpaint.py`, `src/cert_pipeline/flash_composite.py`, `run_batch.py`, `run_batch.sh`, `run_cert.sh`: 수정 전과 **SHA-identical**.
- Protected runtime/model/input/canonical 파일 **6,282개** size/mtime 불변 검사. 이번 작업에서 대형 model 전체 SHA를 새로 계산한 것은 아니다. 기존 migration SHA 증거를 보존하고 변경하지 않았음을 scope/stat 및 작은 경계 파일 hash로 확인했다.
- Canonical 입력 10개(source5+PreProc5)는 이번 CPU 실행 전후 SHA를 검사했다.
- Reference `pipeline`, `VideoOutpainting`, `Upscaler`는 READ ONLY로 사용했다. Source/config/model을 쓰거나 옮기거나 삭제하지 않았다.

구체적 상태: `state/preprocess_reproduction/static_checks.json`, `scope_check.json`, `cpu_tests.log`, `ruff.log`, `diff_*.patch`. 수정 전 bytes는 `state/preprocess_reproduction/before/`, 인덱스는 `before_index.json`에 있다. 기존 migration/source manifests는 이전 snapshot을 나타내므로 덮어써서 현재 Stage0001이 old CRF1 producer와 byte-identical인 것처럼 만들지 않았다.

Historical producer SHA256: `381e55c54645b45aa721d3fbf160923beaae52a2939a4d11a4b8b49a98cdb894`  
Corrected shared preprocess.py SHA256: `c46dd3442cf0a76320d94440f025cd1fed0eca2320d2140ca25c40f83a84484d`

`tools/final_static_validation.py`의 기존 main은 이전 byte-identity 조건을 담은 historical verifier다. 그대로 실행해 과거 migration evidence를 덮어쓰지 않았으며, 현재 Stage0001 acceptance는 새 CPU regression/실측 comparison을 사용한다. Pure schema helper만 재사용했다.

## 8. 문서/history 갱신

- `docs/PIPELINE_EXECUTION_AUDIT.md`: corrected Stage0001 addendum + original audit archive.
- `docs/MIGRATION_DIFF_AUDIT.md`: current correction/diff classification + old migration evidence.
- `docs/HISTORICAL_PREPROCESS_PROVENANCE.md`: current corrected contract + past provenance report.
- `docs/DIRECTORY_LAYOUT.md`: immutable reference / mutable generated / isolated validation 경로.
- `docs/REPRODUCIBILITY_05.md`: 과거 CRF1 FAIL의 시점 표시, 이번 CPU 결과 링크. GPU 재현성으로 바꿔 쓰지 않음.
- `README.md`: 현재 Stage0001 PASS와 별도 GPU 미검증 경계를 표시.

## 9. 남은 검증 경계 / 다음 단계

**이번 목표는 충족했다:** current 8K → corrected shared Stage0001 → historical canonical과 같은 전체 decoded frames.

이 결과로 모델/seed/LongRange/Flash 재현성까지 주장하지 않는다. 05 Outpainting 65f A/B는 이번 작업에서 **실행하지 않았으며**, GPU 작업 금지에 따라 여기서 멈춘다. 새로운 FFmpeg/x265 build나 다른 source로 옮긴 경우에는 producer identity를 확인하고 같은 gate를 다시 검증해야 한다. 이미 결과가 존재하는 validation harness는 자동 덮어쓰지 않고 거부한다.

```text
Full uses same run_preprocess_stage: YES
Quick uses same run_preprocess_stage: YES
Stage-only uses same run_preprocess_stage: YES
Duplicated preprocessing implementation: NO
Outpainting computation: UNCHANGED
FlashVSR computation: UNCHANGED
Checkpoint / models: UNCHANGED
Seeds: UNCHANGED
Queue implementation: UNCHANGED
GPU work: NO
```
