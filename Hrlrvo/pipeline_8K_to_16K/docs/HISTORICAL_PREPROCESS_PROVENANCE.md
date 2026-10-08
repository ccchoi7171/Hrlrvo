# Historical Preprocess Provenance

## Current correction — historical reproduction contract

**Previous certification draft: CRF1.**  
**Corrected historical reproduction contract: CRF18.**  
**Reason: historical PreProc reproduction requirement.** The user explicitly corrected the earlier CRF1 specification; this is an intentional Stage0001 correction, not a hidden change to copied model computation.

The complete historical command is now enforced, not just CRF: input `-threads 8`, `-loglevel error`, one video + one audio with `-map 0:v:0 -map 0:a:0`, Lanczos/pc→pc, `setsar=1,setpts=PTS*2/3,fps=24`, `-fps_mode:v passthrough`, libx265/fast/CRF18/yuv420p10le, `-x265-params log-level=error:pools=8`, pc/bt709/bt709 tags, no color_transfer override, hvc1, atempo1.5/AAC192k, map_metadata0, timescale24576, `-n`, and historical `_PreProc_2K[.partial].mp4` naming. Full/Quick/validation JSON and schema carry the same fixed contract.

Current source: `src/cert_pipeline/preprocess.py:302-376` (`build_ffmpeg_command`), `:200-291` (input/output validation), `:422-473` (strict full video/audio decode), `:522+` (`run_preprocess`). Shared runner: `run_cert.py:918-989` (`run_preprocess_stage`). These anchors describe the corrected workspace, not the archived old package.

### Current Stage0001 control flow

```text
run_cert.sh -> run_cert.py:main()
  -> validate_config() [one PREPROCESS_CONTRACT / JSON/schema]
  -> execute_stages(stage=all or preprocess)
       -> run_preprocess_stage()
            -> preflight_source(): header nb_frames, historical source validation
            -> source SHA + producer_identity(contract, implementation SHA, tool versions)
            -> preprocess.run_preprocess()
                 -> header probe (no source -count_frames)
                 -> historical build_ffmpeg_command() and CPU libx265
                 -> output header/resolution/fps/count/pixel/color/timebase/audio/duration checks
                 -> strict full FFmpeg video/audio decode + count
                 -> atomic output + producer.json/full_decode.json/frame_alignment.json
            -> publish mutable generated artifact + SHA-bound receipt
       -> run_outpaint_stage() [only for all; model code unchanged]
            -> validate_preprocessed() -> selected generated artifact
```

Quick still preprocesses **all source frames**; only model frame limits differ. Startup Stage0001 no longer does the old repeated 8K `-count_frames` scans: it follows historical `nb_frames` handling and performs strict output full-decode verification. Full input SHA and full source downscale/encode still occur. FlashVSR's separate original-source checks are unchanged.

### Canonical reference preservation and effective artifact path

- Immutable imported reference: `data/01_PreProc_2K_1920x1080/<ID>_PreProc_2K.mp4` and its original JSON. Neither this validation nor future fresh Full publishes over that path.
- Normal fresh Full/Quick/stage-only preprocess output: `data/01_PreProc_2K_1920x1080/metadata/<ID>/generated/<ID>_PreProc_2K.mp4` plus receipt.
- `artifacts.preprocess` retains the declared canonical/cache path. `preprocess_output_path()` derives the mutable destination from `artifacts.preprocess_metadata`; `preprocessed_input_path()` prefers that generated artifact, else the validated canonical cache. Both Full and stage-only use this same selector.
- `resolved_config.json.resolved_preprocess_artifact` records the actual selected path. Invalid generated artifacts fail closed, not silently fall back.
- Validation configs explicitly route outputs to `data/validation/preprocess_reproduction/<ID>/`; they never publish normal/canonical outputs.
- Resume requires source identity, full corrected contract, implementation SHA and FFmpeg/FFprobe version identity. CRF1 receipts do not qualify. Stage-only validates historical exact command provenance or corrected generated producer identity.
- Batch queue code remains unchanged: it uses the existing stage-only Outpaint input validation. It has no independent preprocessing implementation and does not automatically create missing preprocessing artifacts; when needed, the shared `--stage preprocess` is the preparation path.

### Verification and historical evidence boundary

**Actual CPU acceptance: PREPROCESS REPRODUCED.** All five videos passed **65/65 first frames and 1952/1952 full RGB24 frames** against their current canonical inputs. All five regenerated MP4 SHA256 values also matched canonical. Evidence is recorded in [PREPROCESS_REPRODUCTION_01_05.md](PREPROCESS_REPRODUCTION_01_05.md) and `data/validation/preprocess_reproduction/summary.json`; this is an actual CPU experiment, not a contract-only inference. GPU/Outpainting/FlashVSR were not executed by this correction task. Outpainting's own CRF1 chunk/retime encoding and Flash/final composite settings are **unchanged**; only Stage0001 changed.

Old migration/static manifests describe the historical snapshot and are preserved, not rewritten to pretend the new producer is byte-identical. Current diffs, static checks and protected-scope evidence are under `state/preprocess_reproduction/`. The old `tools/final_static_validation.py` main encodes the previous byte-identity assumptions and is retained as historical verification code, not used as the corrected Stage0001 acceptance check.

---

## Archived pre-correction report — historical facts, NOT current Stage0001 behavior

The original document bytes are preserved at `../state/preprocess_reproduction/before/docs/HISTORICAL_PREPROCESS_PROVENANCE.md`. Statements below such as “current CRF1”, old source line numbers, copy identity, NO execution, or prior05 gate FAIL describe that earlier audit/migration. They must not override the current correction above. No past evidence has been silently rewritten.

### VERDICT

- **Historical 01–05 PreProc generation command: FOUND EXACTLY**
- **CRF18 evidence: CONFIRMED**
- **Current CRF1 Stage origin: IDENTIFIED**
- **01–05 historical preprocessing same settings: YES** — 아래에서 확인한 현재 canonical 5개 파일의 FFmpeg 인수는 input/output 경로를 제외하고 같다. 추가로 복구한 초기 seed-search용 03의 이전 버전도 같은 CRF18 명령이다.
- **Historical data COPY: PASS / historical preprocessing SEMANTICS: MISMATCH**

핵심 결론: **파일을 잘못 복사한 것이 아니다.** 기존 PreProc은 CRF18로 생성됐지만, 이후 certification pipeline 구축 시 **명시된 CRF1 요구를 적용한 별도의 Stage 0001 코드가 작성**됐다. 새 작업공간은 그 audited CRF1 코드를 그대로 복사했다. 따라서 “historical PreProc을 byte-identical하게 복사했다”와 “8K부터 다시 실행하면 historical PreProc과 동일한 픽셀이 만들어진다”는 서로 다른 주장이고, 후자는 성립하지 않는다.

**중요한 추가 provenance:** 초기 03 seed-search가 사용한 PreProc은 현재 canonical 03과 SHA가 다르다. 이번 조사에서 해당 **이전 03 MP4, 이전 8K, sidecar와 실제 재생성 로그까지 찾아 SHA로 연결**했다. 03은 CRF 변경이 아니라 **원본 8K crop/resize 교체 후 같은 CRF18 명령으로 재생성된 별도 데이터 버전**이다. 이 문제는 certification CRF18 ↔ CRF1 차이와 분리해야 한다.

#### 조사 경계

- 조사 방식: READ-ONLY source/history/provenance 조사 + 기존 파일 SHA256 / FFprobe header·packet·HEVC encoder metadata 읽기.
- 코드/config/기존 영상 수정: **NO**.
- 새 preprocess candidate 생성: **NO**.
- FFmpeg preprocessing / GPU inference / CUDA model load / Outpainting / FlashVSR / queue / smoke test 실행: **NO**.
- 이번 조사에서 전체 pixel decode 또는 framemd5 재생성: **NO**. 이전 검증의 framemd5 텍스트만 다시 비교했다.
- 유일한 새 파일: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/docs/HISTORICAL_PREPROCESS_PROVENANCE.md`.
- 표의 mtime은 **UTC**다. mtime은 생성 시점 자체를 증명하지 않으며, 생성 이력은 command/manifest/session 자료와 별도로 판단한다.

#### 신뢰도 용어

- **CONFIRMED:** 현재 파일 bytes/직접 읽은 metadata/정확한 명령 기록/코드로 확인.
- **STRONGLY SUPPORTED:** 경로·시간·주변 기록이 일치하지만 해당 실행의 input SHA 같은 마지막 연결 증거가 없음.
- **INFERRED:** 간접 근거에 기반한 추론. 확정과 구분.
- **UNKNOWN:** 기록에서 확인하지 못함.
- **NOT RUNTIME-VERIFIED:** 이번 조사에서 새 인코딩 또는 모델 실행으로 검증하지 않은 항목.

### 1. A와 B를 분리한 판단

| 질문 | 결론 | 근거 |
| --- | --- | --- |
| A. 기존 canonical 8K / PreProc이 새 폴더로 그대로 복사됐는가? | PASS / CONFIRMED | 이번 조사에서 10개 source MP4와 10개 copied MP4의 전체 SHA256을 읽어 모두 일치 확인. PreProc sidecar의 input/output SHA도 현재 source/output과 일치. |
| B. 현재 certification Stage 0001은 historical 생성 명령과 동일한가? | MISMATCH / CONFIRMED | historical: -crf 18. audited/new: -crf 1. 추가로 decoder thread/log/audio/temporary-output 정책에 차이가 있음. |
| 복사 PASS이면 새 8K→2K 재생성도 exact인가? | NO | 보존된 데이터 bytes와 다시 계산하는 producer의 조건은 독립적이다. |
| 새 Outpainting 결과 자체가 달라졌다고 실험으로 확인됐는가? | NO | 이전 05 검증은 preprocessing first65 exact-input gate에서 중단됐다. 새 Outpainting/Flash 실행 결과 비교가 아니다. |

Source of truth 경로:
```text
/home/work/ai-graphics_33/Hrlrvo/pipeline
/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K
/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting
/home/work/ai-graphics_33/Hrlrvo/Upscaler
```

아래 `pipeline/...` 및 `pipeline_8K_to_16K/...`는 모두 `/home/work/ai-graphics_33/Hrlrvo` 기준 상대 경로다. Session 증거는 별도 절에 절대 경로를 명시한다.

### 2. Historical Files — 현재 canonical PreProc 01–05

#### 수집 방법과 공통 해석

- SHA256: 전체 파일 bytes를 읽어 계산. bitrate로 유추하지 않았다.
- metadata: FFprobe `-show_streams -show_format -of json`.
- frame count 표의 “reported”는 `nb_frames`이며, 별도로 `-show_packets`에서 video packet 1952개 / unique presentation PTS 1952개를 확인했다.
- sidecar의 `verification.decoded_frames=1952, full_decode=ok`는 **과거 생성 시 전체 decode 검증 기록**이다. 이번에 1952프레임 pixel decode를 반복한 것이 아니다.
- codec level은 FFprobe가 반환한 정수 값을 그대로 적는다.
- `color_transfer`와 `creation_time`은 모든 현재 canonical 파일에서 미보고다. 다른 color 필드가 BT.709라고 해서 transfer 값을 임의 보충하지 않는다.
- track timescale 24576은 FFprobe time_base `1/24576`와 실제 명령 `-video_track_timescale 24576`가 일치한다.
- 출력 파일명 “2K”는 이 프로젝트 명칭이며 실제 해상도는 1920×1080이다.

#### 01: 01_PreProc_2K.mp4

| Field | Observed value |
| --- | --- |
| ID | 01 |
| filename | `01_PreProc_2K.mp4` |
| absolute path | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/01_PreProc_2K.mp4` |
| size bytes | 42673684 |
| SHA256 | `228d3e31e72805798a3ba22a4d0cac47b79f7c1d1d41c7f3ede35498f770a85a` |
| resolution | 1920×1080 |
| fps (avg / r) | 24/1 / 24/1 |
| frame count | reported=1952; video packets=1952; unique PTS=1952; historical decoded_frames=1952 |
| duration seconds (video / container) | 81.333333 / 81.345000 |
| codec / tag | hevc / hvc1 |
| profile / level | Main 10 / 120 |
| pix_fmt | `yuv420p10le` |
| bitrate bps (video / container) | 3991885 / 4196809 |
| encoder metadata (video / muxer) | `Lavc60.31.102 libx265` / `Lavf60.16.100` |
| x265 bitstream metadata | 3.5+1-f0c1022b6, 10bit; rc=crf crf=18.0; CONFIRMED |
| color_range | pc |
| colorspace | bt709 |
| color_primaries | bt709 |
| color_transfer | 미보고 / UNKNOWN |
| time_base | 1/24576 |
| video track timescale | 24576 |
| start_pts / start_time | 0 / 0.000000 |
| audio presence / codec | 1 stream / aac (LC) |
| audio format | 48000 Hz; 2 channels (stereo); time_base=1/48000 |
| audio duration / start_time | 81.345000 / 0.000000 |
| creation_time | 미보고 — stream/format tags에 없음 |
| filesystem mtime UTC | 2026-09-21T15:16:13.962976+00:00 |
| sidecar | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/01_PreProc_2K.json` |
| sidecar SHA256 | `17ac09810dbab66037c12baacfe1d02c6a60f6ca114b0c8c6be49b0d84ee70b4` |
| sidecar mtime UTC | 2026-09-21T15:16:23.522301+00:00 |
| copied destination | `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/01_PreProc_2K.mp4` |
| destination SHA256 | `228d3e31e72805798a3ba22a4d0cac47b79f7c1d1d41c7f3ede35498f770a85a` |
| copy / recorded output SHA match | PASS / PASS |

#### 02: 02_PreProc_2K.mp4

| Field | Observed value |
| --- | --- |
| ID | 02 |
| filename | `02_PreProc_2K.mp4` |
| absolute path | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/02_PreProc_2K.mp4` |
| size bytes | 67910799 |
| SHA256 | `58fa85818926402d6827fc59f52713867794ce6e4ed6e6805a05d7f171eb9ccb` |
| resolution | 1920×1080 |
| fps (avg / r) | 24/1 / 24/1 |
| frame count | reported=1952; video packets=1952; unique PTS=1952; historical decoded_frames=1952 |
| duration seconds (video / container) | 81.333333 / 81.337000 |
| codec / tag | hevc / hvc1 |
| profile / level | Main 10 / 120 |
| pix_fmt | `yuv420p10le` |
| bitrate bps (video / container) | 6478674 / 6679449 |
| encoder metadata (video / muxer) | `Lavc60.31.102 libx265` / `Lavf60.16.100` |
| x265 bitstream metadata | 3.5+1-f0c1022b6, 10bit; rc=crf crf=18.0; CONFIRMED |
| color_range | pc |
| colorspace | bt709 |
| color_primaries | bt709 |
| color_transfer | 미보고 / UNKNOWN |
| time_base | 1/24576 |
| video track timescale | 24576 |
| start_pts / start_time | 0 / 0.000000 |
| audio presence / codec | 1 stream / aac (LC) |
| audio format | 48000 Hz; 2 channels (stereo); time_base=1/48000 |
| audio duration / start_time | 81.337000 / 0.000000 |
| creation_time | 미보고 — stream/format tags에 없음 |
| filesystem mtime UTC | 2026-09-21T15:16:32.934285+00:00 |
| sidecar | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/02_PreProc_2K.json` |
| sidecar SHA256 | `18f131cae0969e74831a1947016aa74577bf576ea0556f9047f4df3d6cca5673` |
| sidecar mtime UTC | 2026-09-21T15:16:41.654251+00:00 |
| copied destination | `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/02_PreProc_2K.mp4` |
| destination SHA256 | `58fa85818926402d6827fc59f52713867794ce6e4ed6e6805a05d7f171eb9ccb` |
| copy / recorded output SHA match | PASS / PASS |

#### 03: 03_PreProc_2K.mp4

| Field | Observed value |
| --- | --- |
| ID | 03 |
| filename | `03_PreProc_2K.mp4` |
| absolute path | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.mp4` |
| size bytes | 65729698 |
| SHA256 | `7b2b12ffb5eb8a9854fc121321b15d8b464ee976ed7830b9762282e01601a296` |
| resolution | 1920×1080 |
| fps (avg / r) | 24/1 / 24/1 |
| frame count | reported=1952; video packets=1952; unique PTS=1952; historical decoded_frames=1952 |
| duration seconds (video / container) | 81.333333 / 81.348000 |
| codec / tag | hevc / hvc1 |
| profile / level | Main 10 / 120 |
| pix_fmt | `yuv420p10le` |
| bitrate bps (video / container) | 6262554 / 6464050 |
| encoder metadata (video / muxer) | `Lavc60.31.102 libx265` / `Lavf60.16.100` |
| x265 bitstream metadata | 3.5+1-f0c1022b6, 10bit; rc=crf crf=18.0; CONFIRMED |
| color_range | pc |
| colorspace | bt709 |
| color_primaries | bt709 |
| color_transfer | 미보고 / UNKNOWN |
| time_base | 1/24576 |
| video track timescale | 24576 |
| start_pts / start_time | 0 / 0.000000 |
| audio presence / codec | 1 stream / aac (LC) |
| audio format | 48000 Hz; 2 channels (stereo); time_base=1/48000 |
| audio duration / start_time | 81.348000 / 0.000000 |
| creation_time | 미보고 — stream/format tags에 없음 |
| filesystem mtime UTC | 2026-09-22T07:36:46.482023+00:00 |
| sidecar | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.json` |
| sidecar SHA256 | `6e6583563f5c71044440ba145d4702323ff5674f13a0957fa1d9183a87ce0e60` |
| sidecar mtime UTC | 2026-09-22T07:36:56.035466+00:00 |
| copied destination | `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/03_PreProc_2K.mp4` |
| destination SHA256 | `7b2b12ffb5eb8a9854fc121321b15d8b464ee976ed7830b9762282e01601a296` |
| copy / recorded output SHA match | PASS / PASS |

#### 04: 04_PreProc_2K.mp4

| Field | Observed value |
| --- | --- |
| ID | 04 |
| filename | `04_PreProc_2K.mp4` |
| absolute path | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/04_PreProc_2K.mp4` |
| size bytes | 144359003 |
| SHA256 | `b671e890942a4f2e8e869f383cb73671fd6549ad79bcc92c965d87dcfbd8f393` |
| resolution | 1920×1080 |
| fps (avg / r) | 24/1 / 24/1 |
| frame count | reported=1952; video packets=1952; unique PTS=1952; historical decoded_frames=1952 |
| duration seconds (video / container) | 81.333333 / 81.346000 |
| codec / tag | hevc / hvc1 |
| profile / level | Main 10 / 120 |
| pix_fmt | `yuv420p10le` |
| bitrate bps (video / container) | 13997280 / 14197035 |
| encoder metadata (video / muxer) | `Lavc60.31.102 libx265` / `Lavf60.16.100` |
| x265 bitstream metadata | 3.5+1-f0c1022b6, 10bit; rc=crf crf=18.0; CONFIRMED |
| color_range | pc |
| colorspace | bt709 |
| color_primaries | bt709 |
| color_transfer | 미보고 / UNKNOWN |
| time_base | 1/24576 |
| video track timescale | 24576 |
| start_pts / start_time | 0 / 0.000000 |
| audio presence / codec | 1 stream / aac (LC) |
| audio format | 48000 Hz; 2 channels (stereo); time_base=1/48000 |
| audio duration / start_time | 81.346000 / 0.000000 |
| creation_time | 미보고 — stream/format tags에 없음 |
| filesystem mtime UTC | 2026-09-21T17:05:39.754654+00:00 |
| sidecar | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/04_PreProc_2K.json` |
| sidecar SHA256 | `e558040a3b6f164a6592b3bf2d78fabfeea253e09e3761c2b1d3d68a76c03b07` |
| sidecar mtime UTC | 2026-09-21T17:05:54.367755+00:00 |
| copied destination | `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/04_PreProc_2K.mp4` |
| destination SHA256 | `b671e890942a4f2e8e869f383cb73671fd6549ad79bcc92c965d87dcfbd8f393` |
| copy / recorded output SHA match | PASS / PASS |

#### 05: 05_PreProc_2K.mp4

| Field | Observed value |
| --- | --- |
| ID | 05 |
| filename | `05_PreProc_2K.mp4` |
| absolute path | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4` |
| size bytes | 81412211 |
| SHA256 | `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400` |
| resolution | 1920×1080 |
| fps (avg / r) | 24/1 / 24/1 |
| frame count | reported=1952; video packets=1952; unique PTS=1952; historical decoded_frames=1952 |
| duration seconds (video / container) | 81.333333 / 81.342000 |
| codec / tag | hevc / hvc1 |
| profile / level | Main 10 / 120 |
| pix_fmt | `yuv420p10le` |
| bitrate bps (video / container) | 7806046 / 8006905 |
| encoder metadata (video / muxer) | `Lavc60.31.102 libx265` / `Lavf60.16.100` |
| x265 bitstream metadata | 3.5+1-f0c1022b6, 10bit; rc=crf crf=18.0; CONFIRMED |
| color_range | pc |
| colorspace | bt709 |
| color_primaries | bt709 |
| color_transfer | 미보고 / UNKNOWN |
| time_base | 1/24576 |
| video track timescale | 24576 |
| start_pts / start_time | 0 / 0.000000 |
| audio presence / codec | 1 stream / aac (LC) |
| audio format | 48000 Hz; 2 channels (stereo); time_base=1/48000 |
| audio duration / start_time | 81.342000 / 0.000000 |
| creation_time | 미보고 — stream/format tags에 없음 |
| filesystem mtime UTC | 2026-09-21T15:16:37.118033+00:00 |
| sidecar | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/05_PreProc_2K.json` |
| sidecar SHA256 | `6abbea76bca6e447c9b8f9d5c74653cda078207f5c49d2de0a495fa5f55df34b` |
| sidecar mtime UTC | 2026-09-21T15:16:47.048090+00:00 |
| copied destination | `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4` |
| destination SHA256 | `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400` |
| copy / recorded output SHA match | PASS / PASS |

### 3. Source 8K 01–05와 1:1 대응

모든 현재 source는 **7680×4320, 16/1 fps, HEVC/Main 10, yuv420p10le, reported 1952f, 122초**다. 이번 FFprobe packet 조사에서도 각각 1952 packet 및 1952 unique presentation PTS를 확인했다.

| ID | filename / absolute path | size bytes | SHA256 | copied destination SHA |
| --- | --- | --- | --- | --- |
| 01 | `01_Input_8K.mp4`<br>`/home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/01_Input_8K.mp4` | 376710794 | `2d73992d09e37442fa7f0821711be0bb69584c6c5683ead99a8ad72ffe218d0c` | `2d73992d09e37442fa7f0821711be0bb69584c6c5683ead99a8ad72ffe218d0c` |
| 02 | `02_Input_8K.mp4`<br>`/home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/02_Input_8K.mp4` | 694119810 | `c192abcb6426be964a9a258c0c06fb22a81317eb8771d46fd81085a2172981b1` | `c192abcb6426be964a9a258c0c06fb22a81317eb8771d46fd81085a2172981b1` |
| 03 | `03_Input_8K.mp4`<br>`/home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/03_Input_8K.mp4` | 773738218 | `d4326b28039cfdc51defca0759138dd38db843f7dfd59ed41c1846d4a73e7bbf` | `d4326b28039cfdc51defca0759138dd38db843f7dfd59ed41c1846d4a73e7bbf` |
| 04 | `04_Input_8K.mp4`<br>`/home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/04_Input_8K.mp4` | 1487029060 | `9341bf821851f87845e0b0d16426d05035574e688536ca587fa4e98fc0a338a1` | `9341bf821851f87845e0b0d16426d05035574e688536ca587fa4e98fc0a338a1` |
| 05 | `05_Input_8K.mp4`<br>`/home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/05_Input_8K.mp4` | 854767123 | `684b400a0057478295e22ddf7aa5caf5b5ad1a4fb77fafdbf9a5c74baa2cb454` | `684b400a0057478295e22ddf7aa5caf5b5ad1a4fb77fafdbf9a5c74baa2cb454` |

| ID | resolution | fps avg/r | frames reported / packet | codec / profile / level | pix_fmt | time_base / timescale | start_pts / start_time | duration video / container |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 7680×4320 | 16/1 / 16/1 | 1952 / 1952 | hevc / Main 10 / 180 | yuv420p10le | 1/16384 / 16384 | 0 / 0.000000 | 122.000000 / 122.000000 |
| 02 | 7680×4320 | 16/1 / 16/1 | 1952 / 1952 | hevc / Main 10 / 180 | yuv420p10le | 1/16384 / 16384 | 0 / 0.000000 | 122.000000 / 122.000000 |
| 03 | 7680×4320 | 16/1 / 16/1 | 1952 / 1952 | hevc / Main 10 / 180 | yuv420p10le | 1/16384 / 16384 | 0 / 0.000000 | 122.000000 / 122.000000 |
| 04 | 7680×4320 | 16/1 / 16/1 | 1952 / 1952 | hevc / Main 10 / 180 | yuv420p10le | 1/16384 / 16384 | 0 / 0.000000 | 122.000000 / 122.000000 |
| 05 | 7680×4320 | 16/1 / 16/1 | 1952 / 1952 | hevc / Main 10 / 180 | yuv420p10le | 1/16384 / 16384 | 0 / 0.000000 | 122.000000 / 122.000000 |

| ID | audio | color range / space / primaries / transfer | bitrate video / container bps | mtime UTC |
| --- | --- | --- | --- | --- |
| 01 | 1× aac/LC, 48000Hz 2ch, 122.000000s | pc / bt709 / bt709 / 미보고 | 24367999 / 24702347 | 2026-09-21T14:05:16.557718+00:00 |
| 02 | 1× aac/LC, 48000Hz 2ch, 122.000000s | pc / bt709 / bt709 / 미보고 | 45195745 / 45516053 | 2026-09-21T14:10:32.639521+00:00 |
| 03 | 1× aac/LC, 48000Hz 2ch, 122.000000s | pc / bt709 / bt709 / 미보고 | 50408729 / 50736932 | 2026-09-22T07:17:29.880146+00:00 |
| 04 | 1× aac/LC, 48000Hz 2ch, 122.000000s | pc / bt709 / bt709 / 미보고 | 97179584 / 97510102 | 2026-09-20T08:03:57.901002+00:00 |
| 05 | 1× aac/LC, 48000Hz 2ch, 122.000000s | pc / bt709 / bt709 / 미보고 | 55723418 / 56050303 | 2026-09-21T14:10:08.734866+00:00 |

위 source도 encoder tags는 `Lavc60.31.102 libx265` / `Lavf60.16.100`이며 creation_time은 미보고다. copied destination은 모두 `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/00_Input_8K_7680x4320/<ID>_Input_8K.mp4`다.

#### Manifest가 연결하는 source → PreProc

각 `<ID>_PreProc_2K.json:3-6`에는 input/output 절대 경로와 두 SHA가 있다.

| ID | 현재 8K SHA ↔ sidecar input_sha256 | 현재 PreProc SHA ↔ sidecar output_sha256 | copy source ↔ destination | 연결 |
| --- | --- | --- | --- | --- |
| 01 | MATCH | MATCH | 8K PASS / PreProc PASS | `01_Input_8K.mp4` → `01_PreProc_2K.mp4` |
| 02 | MATCH | MATCH | 8K PASS / PreProc PASS | `02_Input_8K.mp4` → `02_PreProc_2K.mp4` |
| 03 | MATCH | MATCH | 8K PASS / PreProc PASS | `03_Input_8K.mp4` → `03_PreProc_2K.mp4` |
| 04 | MATCH | MATCH | 8K PASS / PreProc PASS | `04_Input_8K.mp4` → `04_PreProc_2K.mp4` |
| 05 | MATCH | MATCH | 8K PASS / PreProc PASS | `05_Input_8K.mp4` → `05_PreProc_2K.mp4` |

이것은 **현재 파일 버전의 provenance 연결**이다. 과거 다른 시점에 같은 경로에 다른 bytes가 있었는지는 별도 문제이며, 03은 실제로 그 예외가 확인됐다(§8).

### 4. Frame structure / timestamp 조사 — pixel 재생성과 구분

#### 이번에 직접 확인한 packet PTS

현재 8K 5개와 PreProc 5개 모두:

```text
video packets = 1952
present PTS = 1952
unique presentation PTS = 1952
sorted first PTS = 0, 1024, 2048, 3072, 4096, 5120
sorted last PTS = 1992704, 1993728, 1994752, 1995776, 1996800, 1997824
all adjacent presentation PTS deltas = 1024
all packet duration ticks = 1024

Source time_base = 1/16384: t[k] = 1024*k/16384 = k/16
PreProc time_base = 1/24576: t[k] = 1024*k/24576 = k/24
N = 1952
source video duration = 1952/16 = 122 s
PreProc video duration = 1952/24 = 81.333333... s
```

| ID | 8K packet-order PTS first12 | PreProc packet-order PTS first12 |
| --- | --- | --- |
| 01 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 |
| 02 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 |
| 03 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 |
| 04 | 0, 4096, 2048, 1024, 3072, 8192, 6144, 5120, 7168, 12288, 10240, 9216 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 |
| 05 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 | 0, 5120, 3072, 1024, 2048, 4096, 10240, 8192, 6144, 7168, 9216, 15360 |

packet decode 순서의 PTS는 B-frame reorder로 단조 증가하지 않지만, presentation 순서로 정렬한 PTS는 모두1024 간격이다. 모든 파일 첫 DTS는 `-2048, -1024, 0, 1024, ...`다. packet순 PTS 차이를 그 자체로 프레임누락/시간shift로 판단하지 않았다.

#### 무엇이 확인됐고 무엇이 아닌가

- **CONFIRMED:** 현재 파일들의 해상도/fps/header frame count/packet count/PTS 격자와 sidecar input/output SHA가 일치.
- **CONFIRMED:** historical 생성 코드는 원본을 Lanczos downscale하고 `setpts=PTS*2/3,fps=24`를 적용한다.
- **CONFIRMED:** historical sidecar는 각각 1952 decoded frames 및 full_decode=ok를 기록한다.
- **NOT RUNTIME-VERIFIED (이번 조사):** 각 source pixel[k]를 다시 downscale했을 때 historical pixel[k]와 exact인지. 이를 확인할 새 candidate는 만들지 않았다.
- `fps=24`는 실제 resampler다. 현재 CFR 입력은 위 시간축과 맞고 결과 N도 일치하지만, “count 일치”만으로 모든 frame의 내용/순서가 수학적으로 증명된다고 하지 않는다.
- ffprobe packet 순회는 압축 packet/header 읽기이며 `-count_frames` 또는 전체 pixel decode와 구분된다.

### 5. Were all 01–05 generated with the same preprocessing settings?

**YES — CONFIRMED.**

1. 5개 sidecar의 `ffmpeg_command` 배열을 비교했다. input/output path 두 값만 제거하면 **전체 인수 배열이 동일**하다.
2. 실제 MP4의 HEVC encoder SEI에 기록된 **전체 x265 option string이 5개 모두 동일**하다.
3. 모두 FFmpeg/FFprobe `6.1.1-3ubuntu5`, Python `3.11.15`를 기록한다.
4. 모두 `preset fast / crf 18 / yuv420p10le / pools=8 / atempo=1.5 / AAC 192k / video_track_timescale 24576`다.
5. bitrate와 파일크기는 영상별로 다르지만 그 차이로 encoder 설정 차이를 추정하지 않았다.
6. 추가로 찾은 **초기 03 seed-search용 이전 MP4**도 같은 전체 x265 option string, 같은 CRF18 명령을 갖는다.

#### CRF18의 직접 증거

일반적인 `ffprobe tags.encoder=Lavc60.31.102 libx265`만으로는 CRF를 알 수 없다. 이번 판정은 다음 세 가지를 함께 사용했다.

| 증거 | 내용 | 강도 |
| --- | --- | --- |
| 실제 HEVC bitstream metadata | hvcC codec extradata에서 prefix SEI NAL type39 / user_data_unregistered payload에 x265 options가 존재. rc=crf crf=18.0 직접 확인. | CONFIRMED |
| 파일별 생성 sidecar | ffmpeg_command의 -crf 18, crf:18, 실제 파일과 일치하는 output_sha256. | CONFIRMED |
| 실제 생성 Python | src/00to01/preproc_8k_to_2k_24.py:35 및 :124-178의 명령 builder. | CONFIRMED |
| bitrate만으로 CRF 역산 | 사용하지 않음. | 해당 없음 |
| MediaInfo 결과 | 사용하지 않음. | 해당 없음 |

| ID | hvcC bytes / NAL types | SEI 핵심 문자열 | sidecar | 최종 판단 |
| --- | --- | --- | --- | --- |
| 01 | 2459 / 32, 33, 34, 39 | `rc=crf crf=18.0` | `crf:18 / -crf 18` | CONFIRMED |
| 02 | 2459 / 32, 33, 34, 39 | `rc=crf crf=18.0` | `crf:18 / -crf 18` | CONFIRMED |
| 03 | 2459 / 32, 33, 34, 39 | `rc=crf crf=18.0` | `crf:18 / -crf 18` | CONFIRMED |
| 04 | 2459 / 32, 33, 34, 39 | `rc=crf crf=18.0` | `crf:18 / -crf 18` | CONFIRMED |
| 05 | 2459 / 32, 33, 34, 39 | `rc=crf crf=18.0` | `crf:18 / -crf 18` | CONFIRMED |

공통 실제 encoder header 요약:
```text
x265 (build 199) - 3.5+1-f0c1022b6:[Linux][GCC 13.2.0][64 bit] 10bit
frame-threads=3 numa-pools=8
bitdepth=10 input-csp=1 fps=24/1 input-res=1920x1080
rc=crf crf=18.0
sar=1 range=1 colorprim=1 transfer=2 colormatrix=1
```

이는 파일 전체 binary에 문자열 grep을 한 것이 아니다. FFprobe로 제한된 codec extradata를 읽고 HEVC 배열/SEI를 확인했다. source와 PreProc 영상의 encoder 설정을 혼동하지 않는다(예: 이전 03 8K의 CRF16은 2K preprocessing CRF18과 다른 단계).

### 6. Exact historical command

#### 실제 producer / wrapper

```text
pipeline/config/01/01_preproc_2K.sh
pipeline/config/02/02_preproc_2K.sh
pipeline/config/03/03_preproc_2K.sh
pipeline/config/04/04_preproc_2K.sh
pipeline/config/05/05_preproc_2K.sh
  -> pipeline/src/00to01/preproc_8k_to_2k_24.py:main()
  -> preprocess(input_path, output_dir, overwrite)
  -> build_ffmpeg_command(input_path, partial_path, ffmpeg)
  -> subprocess.run(command, check=True)
  -> probe(partial_path)
  -> decode_all_frames(partial_path)
  -> partial_path.replace(output_path)
  -> <ID>_PreProc_2K.json
```

- wrapper `config/<ID>/<ID>_preproc_2K.sh:4-8`: package-relative input을 받아 Python script를 `--output-dir ... --overwrite`로 호출한다.
- historical wrapper 경로는 **config/**(단수), 이후 certification JSON 경로는 **configs/**(복수)다.
- historical script `:22-36`: 7680×4320/16 → 1920×1080/24, CRF18 상수.
- `:124-178`: 정확한 FFmpeg 인수 배열.
- `:181-232`: input 검사, input SHA, subprocess 실행.
- `:233-262`: 출력 header/색공간/fps/timebase/frame count/audio duration 및 전체 decode 검증.
- `:268`: 검증된 partial을 최종 이름으로 publish.
- `:272-323`: 실제 command, source/output SHA 및 검증 결과 JSON 저장.
- `:331-342`: input / output-dir / overwrite만 CLI로 받고 CRF CLI override는 없다.
- seed는 `null`; sidecar `seed_note`는 preprocessing에 random operation이 없다고 명시한다.

**아래 명령은 새로 추정한 명령이 아니라 각 기존 JSON의 `ffmpeg_command` 배열을 shell quoting하여 표시한 것이다. 이번에는 실행하지 않았다.**


#### 01 — `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/01_PreProc_2K.json:10-61`

```bash
/usr/bin/ffmpeg -hide_banner -nostdin -loglevel error -xerror -err_detect explode -threads 8 -i /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/01_Input_8K.mp4 -map 0:v:0 -map 0:a:0 -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' -fps_mode:v passthrough -c:v libx265 -preset fast -crf 18 -pix_fmt yuv420p10le -x265-params log-level=error:pools=8 -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 -af atempo=1.5 -c:a aac -b:a 192k -map_metadata 0 -video_track_timescale 24576 -n /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/01_PreProc_2K.partial.mp4
```

#### 02 — `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/02_PreProc_2K.json:10-61`

```bash
/usr/bin/ffmpeg -hide_banner -nostdin -loglevel error -xerror -err_detect explode -threads 8 -i /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/02_Input_8K.mp4 -map 0:v:0 -map 0:a:0 -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' -fps_mode:v passthrough -c:v libx265 -preset fast -crf 18 -pix_fmt yuv420p10le -x265-params log-level=error:pools=8 -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 -af atempo=1.5 -c:a aac -b:a 192k -map_metadata 0 -video_track_timescale 24576 -n /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/02_PreProc_2K.partial.mp4
```

#### 03 — `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.json:10-61`

```bash
/usr/bin/ffmpeg -hide_banner -nostdin -loglevel error -xerror -err_detect explode -threads 8 -i /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/03_Input_8K.mp4 -map 0:v:0 -map 0:a:0 -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' -fps_mode:v passthrough -c:v libx265 -preset fast -crf 18 -pix_fmt yuv420p10le -x265-params log-level=error:pools=8 -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 -af atempo=1.5 -c:a aac -b:a 192k -map_metadata 0 -video_track_timescale 24576 -n /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.partial.mp4
```

#### 04 — `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/04_PreProc_2K.json:10-61`

```bash
/usr/bin/ffmpeg -hide_banner -nostdin -loglevel error -xerror -err_detect explode -threads 8 -i /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/04_Input_8K.mp4 -map 0:v:0 -map 0:a:0 -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' -fps_mode:v passthrough -c:v libx265 -preset fast -crf 18 -pix_fmt yuv420p10le -x265-params log-level=error:pools=8 -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 -af atempo=1.5 -c:a aac -b:a 192k -map_metadata 0 -video_track_timescale 24576 -n /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/04_PreProc_2K.partial.mp4
```

#### 05 — `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/05_PreProc_2K.json:10-61`

```bash
/usr/bin/ffmpeg -hide_banner -nostdin -loglevel error -xerror -err_detect explode -threads 8 -i /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/05_Input_8K.mp4 -map 0:v:0 -map 0:a:0 -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' -fps_mode:v passthrough -c:v libx265 -preset fast -crf 18 -pix_fmt yuv420p10le -x265-params log-level=error:pools=8 -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 -af atempo=1.5 -c:a aac -b:a 192k -map_metadata 0 -video_track_timescale 24576 -n /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/05_PreProc_2K.partial.mp4
```

#### Partial 이름과 최종 이름이 다른 이유

FFmpeg 인수는 `<ID>_PreProc_2K.partial.mp4`에 쓰도록 되어 있다. Python이 검증 후 `<ID>_PreProc_2K.mp4`로 rename하고 JSON에는 최종 경로와 그 SHA를 기록한다. 따라서 command의 output과 sidecar의 `output` 경로 차이는 이 절차에 의한 것이다.

FFmpeg의 `-n`과 wrapper의 `--overwrite`도 서로 다른 계층이다. 전자는 기존 partial 덮어쓰기를 막고, 후자는 검증 후 최종 파일 교체를 허용한다. **이번 조사에서 해당 동작을 실행한 것은 아니다.**

#### Historical script/config source

| 항목 | 확인 내용 |
| --- | --- |
| path | `/home/work/ai-graphics_33/Hrlrvo/pipeline/src/00to01/preproc_8k_to_2k_24.py` |
| 핵심 lines | 29–36, 124–178, 181–328, 331–348 |
| script SHA256 | `381e55c54645b45aa721d3fbf160923beaae52a2939a4d11a4b8b49a98cdb894` |
| git commit | `c6e32618b76bad264b994cc4fabcdc47d6570375` |
| config | 해당 Python 상수 + 각 video shell wrapper. 당시 CRF18을 공급한 별도 YAML/JSON config가 아니라 생성 후 JSON sidecar에 실행 조건을 기록하는 구조. |
| 실제 outer shell invocation | wrapper와 Python 경로는 확인. 모든 ID에 대해 사용자가 당시 어느 shell 입력으로 wrapper/직접 Python을 시작했는지는 별도의 shell-history 증거가 없음. FFmpeg 실제 argv는 각 sidecar에서 정확히 복구됨. |

### 7. Historical 명령의 별도 terminal 기록 / 관련 프로젝트 구분

#### 03 재생성 실제 로그

`/home/work/ai-graphics_33/Hrlrvo/crop_bbox_preview_03/preproc03_refresh_20260922T073613Z.log:1-4`:

```text
Processing: /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/03_Input_8K.mp4
Output: /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.mp4
Command: /usr/bin/ffmpeg -hide_banner -nostdin -loglevel error -xerror -err_detect explode -threads 8 -i /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/03_Input_8K.mp4 -map 0:v:0 -map 0:a:0 -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' -fps_mode:v passthrough -c:v libx265 -preset fast -crf 18 -pix_fmt yuv420p10le -x265-params log-level=error:pools=8 -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 -af atempo=1.5 -c:a aac -b:a 192k -map_metadata 0 -video_track_timescale 24576 -n /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.partial.mp4
Verified: /home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.mp4 (1920x1080, 24fps, 1952 unchanged frames)
```

이 로그의 “Command:”는 current 03 sidecar의 정확한 argv와 일치하고, 마지막 “Verified:”는 실제 생성 script의 `preprocess():324-327` 출력과 일치한다.

#### 관련되어 보이지만 다른 산출물

- `Data/kbs_backup/01_Resize_2K_1920_1080/<ID>_Resize_2K.json`: 별도 **16fps resize** 계열이다. 이것을 `PreProc_2K/24fps` 생성 명령으로 대체하지 않았다.
- `KBS-OutPaint_Data/mainTest/cropped_16x9/2m02s_16fps/pipeline_manifest.json:144-145` 등의 Lanczos/CRF18 기록은 **16fps + audio copy**인 다른 변환이다. canonical PreProc의 `atempo=1.5 / timescale24576`와 구분한다.
- `Pipeline_8K16K/environment/logs/`에는 libx264/CRF16 및 dry-run/다른 경로 관련 명령이 있다. 현재 historical MP4의 SHA/sidecar 연결을 대신하는 근거가 아니다.
- `VideoOutpainting/main/tools/` 및 실험 metadata는 PreProc을 **소비하는 코드/기록**이다. FlashVSR/Upscaler를 PreProc 생성자로 판단할 근거는 발견되지 않았다.

#### 검색 범위와 한계

`/home/work/ai-graphics_33/Hrlrvo` 전체에서 ignore를 우회한 read-only 관련 키워드 검색을 수행했다. 대상은 `.sh/.py/.yaml/.yml/.json/.md/.txt/.log/.out` text이며, `.git` binary/object 및 모델 weight/video binary는 grep하지 않았다. 해당 bounded 검색은 파일당 5MB 이하를 대상으로 471개의 관련 text 파일을 찾았다. 이후 sidecar, archive, script, run context, audit, session evidence를 구체적으로 읽었다. 이것은 시스템 전체의 무차별 검색이나 모든 모델 asset hash 작업이 아니다.

- `/home/work/ai-graphics_33/.bash_history`: 존재하지만 이번 preprocessing 관련 keyword hit 0.
- `/home/work/ai-graphics_33/.zsh_history`: 없음.
- 조사한 관련 파일들에서 별도 tmux capture가 결정적 생성 provenance로 확인되지는 않았다. 실제 `preproc03_refresh_...log` 및 sidecar가 충분한 직접 증거다.
- Codex session JSONL은 별도 제한된 provenance 조사로 해당 user requirement / Add File 기록을 읽었다(§10). “검색에서 못 찾음”을 “그런 기록은 어디에도 없다”로 확대하지 않았다.

### 8. 실제 seed-search 소비 이력과 03의 두 버전

#### Seed-search run context와 현재 bytes 대조

Source: `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_seedSearch_4K_3840x2160/run_context.json`; created_at=`2026-09-21T18:56:43.129847+00:00`.

| ID | seed-search recorded input SHA256 | 현재 canonical PreProc SHA256 | 결론 |
| --- | --- | --- | --- |
| 01 | `228d3e31e72805798a3ba22a4d0cac47b79f7c1d1d41c7f3ede35498f770a85a` | `228d3e31e72805798a3ba22a4d0cac47b79f7c1d1d41c7f3ede35498f770a85a` | MATCH / CONFIRMED |
| 02 | `58fa85818926402d6827fc59f52713867794ce6e4ed6e6805a05d7f171eb9ccb` | `58fa85818926402d6827fc59f52713867794ce6e4ed6e6805a05d7f171eb9ccb` | MATCH / CONFIRMED |
| 03 | `000f79ec66ab88aaca32dd008f688ddaea4aced5ceefdc5a4ab1f9bd121b6c67` | `7b2b12ffb5eb8a9854fc121321b15d8b464ee976ed7830b9762282e01601a296` | DIFFERENT VERSION / CONFIRMED |
| 04 | `b671e890942a4f2e8e869f383cb73671fd6549ad79bcc92c965d87dcfbd8f393` | `b671e890942a4f2e8e869f383cb73671fd6549ad79bcc92c965d87dcfbd8f393` | MATCH / CONFIRMED |
| 05 | `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400` | `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400` | MATCH / CONFIRMED |

해당 `_runs/<ID>_<seed>/metadata.json` 기록도 대조했다. 03의 10개 seed-search metadata 모두 이전 SHA `000f79ec66ab88aaca32dd008f688ddaea4aced5ceefdc5a4ab1f9bd121b6c67`를 기록한다. 예: `_runs/03_17473905/metadata.json:96-97,262`. 01/02/04/05는 확인한 seed-search SHA가 현재 canonical과 일치한다.

#### 찾은 이전 03 원본/PreProc — 더 이상 “파일 미발견”이 아님

| 역할 | 현재 보존 경로 | size bytes | 이번에 계산한 SHA256 |
| --- | --- | --- | --- |
| 이전 03 8K source | `/home/work/ai-graphics_33/Hrlrvo/crop_bbox_preview_03/03_Input_8K_before_crop_20260922T070256Z.mp4` | 643769373 | `5eb5cce4d95c727ca53a6078e0cc1eb29c5dc2dcabfa568bc3fad3c18d945d8b` |
| 이전 03 seed-search PreProc | `/home/work/ai-graphics_33/Hrlrvo/crop_bbox_preview_03/03_PreProc_2K_before_crop_20260922T073544Z.mp4` | 64319655 | `000f79ec66ab88aaca32dd008f688ddaea4aced5ceefdc5a4ab1f9bd121b6c67` |

- 이전 PreProc SHA는 위 seed-search `input_sha256`와 **exact match**.
- 이전 sidecar `crop_bbox_preview_03/03_PreProc_2K_before_crop_20260922T073544Z.json:5-6`의 input/output SHA가 두 backup MP4와 각각 일치한다.
- 이전 sidecar도 `-crf 18`를 기록하며, backup MP4의 실제 x265 SEI도 `crf=18.0`다.
- 이전/현재 03 sidecar의 FFmpeg argv는 **완전히 동일**하다. 경로 문자열은 그대로였으나 그 경로의 input bytes가 교체됐다.
- 이전03과 현재01–05 전체 x265 option string도 동일하다.
- 따라서 “03은 다른 CRF라서 SHA가 달라졌다”가 아니다.

#### 03 변경 이력

`crop_bbox_preview_03/replace_03_input_20260922T070256Z.json:6-11,17-24,33-42,62-66`은 다음 실제 작업을 기록한다:

```text
2026-09-22T07:02:57Z ～ 2026-09-22T07:19:51Z
operation = crop_bottom_anchored_then_upscale_to_original_resolution
crop_xywh = [224, 252, 7232, 4068]
scale_output = [7680, 4320]
8K replacement video = libx265 / preset fast / CRF16 / yuv420p10le
audio = stream_copy

old 8K SHA = 5eb5cce4d95c727ca53a6078e0cc1eb29c5dc2dcabfa568bc3fad3c18d945d8b
new 8K SHA = d4326b28039cfdc51defca0759138dd38db843f7dfd59ed41c1846d4a73e7bbf

then the same historical CRF18 PreProc command ran again:
old PreProc SHA = 000f79ec66ab88aaca32dd008f688ddaea4aced5ceefdc5a4ab1f9bd121b6c67
new PreProc SHA = 7b2b12ffb5eb8a9854fc121321b15d8b464ee976ed7830b9762282e01601a296
```

변경된 8K SHA는 현재 canonical 03 8K의 실제 SHA와 맞는다. 현재 03 PreProc mtime은 2026-09-22T07:36:46Z이며 재생성 로그/sidecar와 연결된다. **이 교체는 과거 기록이다. 이번 조사에서 교체·crop·재인코딩을 하지 않았다.**

#### 03 Outpainting full-check는 어디까지 확정 가능한가

- `pipeline/data/03_fullCheck_65f_30step/_runs/03_33333333/metadata.json:49,70`은 같은 PreProc 경로와 시작 시간 `2026-09-22T07:44:09.628311+00:00`를 기록한다. 현재 파일 재생성 이후다.
- 해당 metadata에는 input SHA가 없다. 따라서 현재 03 bytes를 소비했다는 것은 경로/시점상 **STRONGLY SUPPORTED**, seed-search처럼 SHA로 확정한 것은 아니다.
- `_runs/03_33333333_old/metadata.json:49,70`은 `2026-09-22T03:45:38.061079+00:00` 시작 기록이며 crop/refresh 이전이다. 이 metadata도 input SHA가 없다.
- 모든 과거 실험을 “현재 같은 filename이므로 같은 bytes”라고 합치지 않는다. baseline을 선택할 때 **실험 시점과 input SHA/version**을 고정해야 한다.

### 9. Current certification Stage 0001

#### Path / current command / origin

- audited: `/home/work/ai-graphics_33/Hrlrvo/pipeline/src/cert_pipeline/preprocess.py`
- new: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/src/cert_pipeline/preprocess.py`
- 두 파일 SHA256 동일: `355f467e13855ee8adf764082b2e2c002e940c4bb5a5a46f4abee60a9d4ac4a0`
- `:31-34`의 VIDEO_FILTER는 historical과 동일.
- `:38`은 **VIDEO_CRF = 1**.
- 실제 builder는 `build_ffmpeg_command():243-314`.
- audited `run_cert.py:137-149`의 config validation 역시 `crf:1`을 요구한다.
- audited/new `configs/certification/05.json:14-29`도 `crf:1`이다. 같은 조건이 certification/quick 01–05에 사용된다.
- 이 모듈을 이번 조사에서 import하거나 실행하지 않았다. 아래는 source에서 읽은 command template이다.

Audio가 있는 현재 5개 입력의 실제 builder template:
```bash
<ffmpeg> -hide_banner -nostdin -loglevel info -xerror -err_detect explode \
  -i <8K_input> -map 0:v:0 -map 0:a:0 \
  -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' \
  -fps_mode:v passthrough -c:v libx265 -preset fast -crf 1 -pix_fmt yuv420p10le \
  -x265-params log-level=error:pools=8 \
  -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1 \
  -af atempo=1.5 -c:a aac -b:a 192k \
  -map_metadata 0 -video_track_timescale 24576 -y <temporary_output>
```

#### 정확한 차이 — 압축 조건과 orchestration 분리

| 항목 | historical producer | audited/new certification | 의미 |
| --- | --- | --- | --- |
| CRF | 18 | 1 | 실제 lossy encode 조건 불일치. pixel identity에 직접 관련. |
| scale / range / setsar | Lanczos / pc→pc / setsar=1 | 동일 | downscale filter 자체는 일치. |
| time filter | setpts=PTS*2/3,fps=24 | 동일 | 16→24 시간 처리는 일치. |
| codec / preset / pix_fmt | libx265 / fast / yuv420p10le | 동일 | codec/preset/pixel format은 동일. CRF만 별도 차이. |
| input decoder threads | -threads 8 명시 | 명시 없음 | x265 pools=8과 별개. 단독 pixel 차이의 원인으로 확정하지 않음. |
| x265 params | log-level=error:pools=8 | 동일 | encoder pool 설정 동일. |
| loglevel | error | info | 로그 정책 차이. pixel 차이의 원인으로 취급하지 않음. |
| audio | 1 video + 1 audio 필수 / map0:a:0 | has_audio이면 map0:a:0 | 현재5개는 모두 audio 있음. 일반 입력 계약은 다름. |
| audio encode | atempo=1.5 / AAC192k | 동일 (audio가 있을 때) | 원본 audio를 단순 copy하는 처리가 아님. |
| output collision | -n / fixed .partial.mp4; Python --overwrite | -y / PID temporary output | 작업 파일 정책 차이. |
| frame verification | header count + full FFmpeg decode | ffprobe -count_frames 등 / certification metadata | 검사 구현을 새롭게 통합. historical script의 완전 copy가 아님. |
| paths / manifests | flat PreProc MP4+sidecar | stage-oriented artifacts / certification config | NEW ORCHESTRATION. |

CRF가 다르다는 점은 확정했다. 하지만 추가 encode로 각 option의 영향을 개별 분리한 실험은 하지 않았다. 따라서 “CRF 한 곳만18로 바꾸면 반드시 bit-exact”라고 주장하지 않는다.

### 10. Current CRF1 origin — 생성 시점의 직접 증거

#### Session 식별자

- S1: `/home/work/ai-graphics_33/.codex/sessions/2026/09/23/rollout-2026-09-23T17-58-37-01a0cd7d-0405-7732-8859-ab85cdd14bd9.jsonl`
- S2: `/home/work/ai-graphics_33/.codex/sessions/2026/09/23/rollout-2026-09-23T22-01-58-01a0ce5b-ce8f-7e21-a7f4-182f9f9b0b6e.jsonl`

JSONL의 해당 행에서 role/type/timestamp와 관련 content만 확인했다. unrelated shell history나 session 내용을 보고서에 복제하지 않았다.

| UTC time | 증거 위치 | 확인 내용 |
| --- | --- | --- |
| 2026-09-23T11:26:40.567Z | S1:74, response_item, role=user | 사용자 preprocessing 요구에 명시적으로 “HEVC libx265, 10-bit yuv420p10le, CRF 1”. 동일 문맥에 Lanczos, setpts=PTS*2/3,fps=24, atempo=1.5가 있음. |
| 2026-09-23T12:57:35.912Z | S1:292 | 기존 historical script를 읽은 tool output에 VIDEO_CRF = 18. certification 구현 전 이미18이 읽힌 기록. |
| 2026-09-23T13:01:57.901Z | S1:424 | certification preprocess_stage 구현 작업 시작 기록. |
| 2026-09-23T13:05:47.858Z | S2:48 | Add File: pipeline/src/cert_pipeline/preprocess.py 최초 patch 안에 VIDEO_CRF = 1. |
| 2026-09-23T13:05:49.488Z | S2:50 | 동일 file change 완료 기록에도 VIDEO_CRF = 1. |
| 2026-09-23T13:06:57.337Z | S2:81 | 완료 설명에 기존 Lanczos/same-N retime을 유지하면서 libx265 10-bit CRF1을 사용했다고 명시. |

이 순서가 증명하는 내용:

1. CRF1은 새 certification 모듈의 **최초 생성 시부터** 들어 있었다.
2. historical CRF18 script를 해당 파일 안에서18→1로 바꾼 이력이 아니라 **별도 새 모듈을 CRF1으로 작성**한 것이다.
3. CRF1은 개발자가 근거 없이 임의로 “품질 보존”을 위해 고른 값이라고 볼 수 없다. 당시 사용자 요구의 정확한 값과 일치한다.
4. 그러나 **왜 최초 요구에서 CRF1을 선택했는지**, historical CRF18 input과의 seed-exact reproduction 차이를 당시 명시적으로 승인/합의했는지는 기록으로 확정하지 못한다. 이는 **UNKNOWN**이다.
5. 따라서 “사용자가 historical seed 결과 변경까지 의도했다”는 해석을 붙이지 않는다. 요구된 encoder 값의 출처와 그 선택의 의도는 별개다.

### 11. Git 및 파일 history

READ-ONLY Git 조사 결과:

- 네 보호 경로의 Git top-level은 모두 `/home/work/ai-graphics_33/Hrlrvo`다. 각각 독립된 repository history가 있는 구조가 아니다.
- 확인한 저장소의 유일한 commit:
  `c6e32618b76bad264b994cc4fabcdc47d6570375`
  / `2026-09-23T15:22:53+09:00`
  / `Backup HRLrvo pipeline suite code and docs`.
- historical `pipeline/src/00to01/preproc_8k_to_2k_24.py`는 이 backup commit에 이미 CRF18로 포함된다. 해당 상수에 대한 blame도 최초 snapshot commit이다.
- historical script의 현재 Git diff는 비어 있다.
- `pipeline/src/cert_pipeline/preprocess.py`는 현재 Git status에서 **untracked (??)**다. 따라서 Git log/blame만으로 생성 이전 이력을 찾을 수 없다.
- 새 작업공간은 해당 Git history로 추적되는 별도 committed tree가 아니다.
- **Git에 CRF18→CRF1 change commit은 없다.** 이것은 “변경 출처를 알 수 없다”는 뜻이 아니다. 최초 Add File session 기록이 더 직접적인 증거다.
- backup commit 날짜를 historical 영상/코드의 최초 생성 날짜로 오인하지 않았다.
- Git index 갱신을 피하도록 optional lock을 끄고 읽기 명령만 사용했다. checkout/reset/clean/revert/commit을 실행하지 않았다.

코드 SHA:
```text
/home/work/ai-graphics_33/Hrlrvo/pipeline/src/00to01/preproc_8k_to_2k_24.py 381e55c54645b45aa721d3fbf160923beaae52a2939a4d11a4b8b49a98cdb894
/home/work/ai-graphics_33/Hrlrvo/pipeline/src/cert_pipeline/preprocess.py 355f467e13855ee8adf764082b2e2c002e940c4bb5a5a46f4abee60a9d4ac4a0
/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/src/cert_pipeline/preprocess.py 355f467e13855ee8adf764082b2e2c002e940c4bb5a5a46f4abee60a9d4ac4a0
/home/work/ai-graphics_33/Hrlrvo/pipeline/src/manifests/source_manifest.json 1d165ef72fa6f69c711f15fb970961442d74d0e27fcfff00d01e9b5ad5783968
/home/work/ai-graphics_33/Hrlrvo/pipeline/src/manifests/model_manifest.json 244479e7a4ab96e847ee951de9fa801d54f2044ce1346c494328f52b94293dd6
```

### 12. source_manifest / model_manifest / audit 교차검사

| 증거 | 확인 결과 | 해석 |
| --- | --- | --- |
| pipeline/PIPELINE_EXECUTION_AUDIT.md:215-231 | Stage0001 CRF1 command 명시. | 현재 certification 조건에 대한 기록. |
| pipeline/PIPELINE_EXECUTION_AUDIT.md:831-842, 특히841 | preprocess.py는 NEW ORCHESTRATION / ARTIFACT WRAPPER. | historical producer와 byte-identical하다는 분류가 아님. |
| pipeline/DIRECTORY_LAYOUT.md:39-40 | certification Lanczos/CRF1 표기. | historical file producer provenance와 구분. |
| pipeline_8K_to_16K/docs/MIGRATION_DIFF_AUDIT.md:34-51 | audited preprocess.py → new preprocess.py BYTE IDENTICAL. | 복사 기준은 historical CRF18 script가 아니라 이미 CRF1인 audited module. |
| pipeline_8K_to_16K/docs/REPRODUCIBILITY_05.md:109-122 | historical CRF18와 audited CRF1 입력 차이 기록. | 이전 실제05 first65 mismatch와 연결. |
| pipeline_8K_to_16K/docs/DIRECTORY_LAYOUT.md:12-13 | copied historical18 / fresh audited1 구분. | 두 input producer history가 공존. |
| src/manifests/source_manifest.json | 6,235 entries; cert_pipeline/preprocess.py에 대한 source-copy entry 없음. | 이 manifest로 preprocessor가 historical script에서 byte-copy됐다고 주장할 수 없음. |
| src/manifests/model_manifest.json | copied model/weights provenance. | video preprocessing encoder 설정의 동일성 증명이 아님. |
| src/manifests/package_manifest.json | packaged preprocess.py 파일 hash 포함. | package 존재/무결성 근거이지 historical producer byte identity 근거는 아님. |
| state/migration/copy_integrity.jsonl, copy_summary.json, final_integrity.json | 6,332 copied files / 83,429,041,182 bytes; model/runtime copy hash PASS. | source/model copy와 preprocessing numerical identity를 분리. |

old/new source_manifest 및 model_manifest는 보존된 동일 파일이다. 대형 weights를 이번 provenance 조사에서 다시 모두 hash하거나 load하지 않았다. 모델/runtime COPY PASS는 실제 과거 source/destination SHA inventory와 final_integrity 기록을 읽어 확인한 범위이며, GPU numerical equivalence를 의미하지 않는다.

### 13. Why CRF differs

#### 선택지별 판정

| 가능한 설명 | 판정 | 근거 |
| --- | --- | --- |
| documented intentional change | 일부만 해당 | certification에 CRF1을 요구한 문서/메시지는 존재. historical data를18→1로 바꾸자는 승인 또는 seed 결과 차이를 수용한 합의까지 확인된 것은 아님. |
| copied from another implementation | CRF1의 주된 출처라는 근거 없음 | historical producer는18이고 새 certification Add File은1. 기록된 사용자 명세와 일치. |
| newly chosen during certification pipeline construction | CONFIRMED — 가장 정확한 분류 | historical script 동작을 참고한 NEW ORCHESTRATION에서 명시된CRF1을 처음부터 사용. |
| historical command was misunderstood | 근거 없음 | 구현 전에18상수가 읽혔고 historical command/SEI도18. bitrate로 잘못 추측한 문제가 아님. |
| unknown | 선택의 상위 동기만 UNKNOWN | CRF1 값의 코드 출처는 IDENTIFIED. 왜 사용자가 원래1을 요구했는지와 old-seed exactness 충돌 인지는 확인 안 됨. |

정확한 원인 흐름:

```text
Historical seed input producer
  src/00to01/preproc_8k_to_2k_24.py
  VIDEO_CRF = 18
  -> canonical PreProc data + SHA-linked sidecars

Later certification specification
  explicitly requests libx265 / 10-bit / CRF1
  -> NEW src/cert_pipeline/preprocess.py
  -> VIDEO_CRF = 1 from the first Add File

Migration to pipeline_8K_to_16K
  -> historical PreProc files copied exactly (CRF18 bytes remain)
  -> audited preprocessing module copied exactly (CRF1 code remains)

Result:
  same copied file != same regenerated producer output
  data copy PASS + regeneration semantics MISMATCH
```

즉 새 폴더 이관 중 CRF18이 CRF1로 몰래 바뀐 것이 아니다. **두 값은 audited pipeline 시점에 이미 서로 다른 계층에서 공존했다.**


### 14. 05 기존 검증이 실제로 비교한 것

이번 조사는 아래 기존 검증 기록과 framemd5 텍스트를 읽었을 뿐 새로운 decode/encode를 하지 않았다.

| 역할 | 비교 파일 | 기록된 SHA256 |
| --- | --- | --- |
| historical exact05 | /home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4 | 751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400 |
| audited Stage0001 regenerated05 | /home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/validation/05_preprocess_from_8k/05_PreProc_2K.mp4 | 46df0bee9b675809986e24a572268500775d5a04356c2d28285ce6c693e286dc |

증거:
- `pipeline_8K_to_16K/data/validation/reproducibility_05.json`
- `data/validation/05_old_first65_rgb24.framemd5`
- `data/validation/05_new_first65_rgb24.framemd5`
- `docs/REPRODUCIBILITY_05.md:78-120`

기존 text의 frame-payload MD5를 이번에 다시 비교한 결과:
```text
old RGB24 frame hashes = 65
new RGB24 frame hashes = 65
exact matches = 0/65
first mismatch (zero based) = 0
PREPROCESS_05_FIRST65_DECODED_IDENTITY = FAIL
outpaint_reproduction = STOPPED_PREPROCESS_IDENTITY_FAIL
```

이것은 **2K preprocessing input의 차이**다. 새65f 4K Outpainting 결과 또는 FlashVSR 결과의 차이를 실험으로 입증한 기록이 아니다. 시각적으로 얼마나 큰 차이인지도 위 exact-hash 결과만으로 정량화하지 않는다.

### 15. Was "copy exactly" violated?

#### 1. Historical video/data COPY: PASS

이번 조사에서 canonical source와 새 폴더의 01–05 8K 및 PreProc SHA를 직접 재계산해 모두 일치 확인했다. 과거 CRF18 파일이 migration 과정에서 CRF1로 재인코딩되거나 훼손된 증거는 없다.

다만 “historical”의 기준 시점은 고정해야 한다. **현재 canonical03을 정확히 복사한 것**은 맞지만, **초기03 seed-search input을 복사한 것**과는 다르다. 초기 버전은 §8의 backup에서 복구됐다. 이것은 copy operation 실패가 아니라 baseline version 선택 문제다.

#### 2. Model/runtime COPY: PASS (기존 migration SHA evidence 범위)

`state/migration/copy_integrity.jsonl` 및 `final_integrity.json`:
- `src/video_outpainting/`: 286 copied files
- `src/flashvsr/`: 6 copied files
- `src/models/`: 5,975 copied files
- 원본/대상 SHA mismatch 없음으로 기록.
- preprocessing module old/new SHA는 이번에도 직접 확인했다.
- 별도 Outpaint artifact 경로 adapter의 두 줄 변경은 migration diff에 명시돼 있으며 model computation copy와 구분된다.
- 이번 조사는 전체 대형 weights의 fresh full rehash나 GPU 계산 동등성 실험을 수행하지 않았다.

#### 3. Historical preprocessing SEMANTICS: MISMATCH

역사적 producer와 certification producer의 CRF가 다르다. **모델/runtime을 보존했다는 주장이, 그 모델에 들어가는 encoded input pixels까지 historical과 동일하다는 주장으로 확대되면 잘못**이다.

따라서 요구가 “코드/데이터 copy 보존”이면 그 범위의 PASS는 유지된다. 요구가 “8K부터 다시 시작해 과거 seed 입력을 exact 재현”이면 현 상태는 PASS가 아니다. 이것이 이전05 gate FAIL을 해석하는 정확한 범위다.

### 16. Recommended next action — 이번에는 실행하지 않음

1. **Baseline을 파일명 대신 SHA+version으로 먼저 고정한다.**
   - 01/02/04/05 seed-search는 현재 canonical SHA와 연결된다.
   - 03 초기 seed-search 재현에는 §8의 이전8K/이전PreProc pair가 필요하다.
   - 현재03 crop 이후 full-check를 baseline으로 삼을 경우 그 실험 metadata와 입력 버전을 별도로 명시한다.
2. 목표가 **과거 seed 탐색 입력의 exact 재현**이라면, 다음 변경 계획의 기준은 단순히 “CRF1 상수만18로 수정”이 아니라 **복구한 historical command 전체와 동일 source bytes / FFmpeg+x265 버전 / pixel-format / timestamp / audio 정책**이어야 한다.
3. CRF1을 계속 공인인증 조건으로 사용할 경우, 그것은 historical CRF18와 **다른 입력 생성 조건**이라고 명확히 기록하고 해당 입력에서 다시 seed 결과를 검증해야 한다. 기존 seed 결과가 자동으로 유지된다고 선언하지 않는다.
4. 새 producer를 적용하는 작업은 별도 변경 단계로 남긴다. 기존 canonical PreProc/backup은 overwrite하지 않고, validation 경로에만 future candidate를 생성하는 계획이 안전하다.
5. 향후 승인된 CPU 검증에서 **05 first65 decoded-frame exact match**를 먼저 통과시킨 뒤에만 제한된 Outpainting 재현성 검증으로 진행한다. 이 보고서는 그 실행을 수행하지 않았으며 GPU gate를 임의로 해제하지 않는다.
6. 다음 문서/manifest에서는 “producer code identity”, “copied artifact identity”, “experiment input identity”를 분리해 기록한다. source_manifest의 model copy PASS로 preprocessing pixel exactness를 대신하지 않는다.

#### 이번 조사에서 하지 않은 일

```text
Existing 01-05 PreProc files altered: NO
Existing 8K files altered: NO
Existing pipeline/reference code altered: NO
Config altered: NO
CRF1 -> CRF18 modification: NO
Candidate preprocessing generated: NO
FFmpeg preprocessing executed: NO
Full-frame pixel decode rerun: NO
GPU/CUDA model load: NO
Outpainting executed: NO
FlashVSR executed: NO
Queue executed: NO
New file: docs/HISTORICAL_PREPROCESS_PROVENANCE.md only
```

### Appendix A. 관련 sidecar/metadata 증거 색인

| ID | PreProc sidecar absolute path | FFmpeg command | CRF | verification |
| --- | --- | --- | --- | --- |
| 01 | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/01_PreProc_2K.json` | :10-61 | :70-71 | :77-100 |
| 02 | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/02_PreProc_2K.json` | :10-61 | :70-71 | :77-100 |
| 03 | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.json` | :10-61 | :70-71 | :77-100 |
| 04 | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/04_PreProc_2K.json` | :10-61 | :70-71 | :77-100 |
| 05 | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/05_PreProc_2K.json` | :10-61 | :70-71 | :77-100 |

| 추가 증거 | 위치 |
| --- | --- |
| historical command builder | `/home/work/ai-graphics_33/Hrlrvo/pipeline/src/00to01/preproc_8k_to_2k_24.py:124-178` |
| audited command builder | `/home/work/ai-graphics_33/Hrlrvo/pipeline/src/cert_pipeline/preprocess.py:243-314` |
| new byte-identical builder | `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/src/cert_pipeline/preprocess.py:243-314` |
| early03 seed input hash | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_seedSearch_4K_3840x2160/run_context.json:52-68` |
| early03 actual run metadata | `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_seedSearch_4K_3840x2160/_runs/03_17473905/metadata.json:96-97,262` |
| old03 sidecar + original-input SHA | `/home/work/ai-graphics_33/Hrlrvo/crop_bbox_preview_03/03_PreProc_2K_before_crop_20260922T073544Z.json:5-6,10-71` |
| 03 input replacement provenance | `/home/work/ai-graphics_33/Hrlrvo/crop_bbox_preview_03/replace_03_input_20260922T070256Z.json:6-11,23-42` |
| 03 exact rerun command / success | `/home/work/ai-graphics_33/Hrlrvo/crop_bbox_preview_03/preproc03_refresh_20260922T073613Z.log:1-4` |
| CRF1 requirement | `/home/work/ai-graphics_33/.codex/sessions/2026/09/23/rollout-2026-09-23T17-58-37-01a0cd7d-0405-7732-8859-ab85cdd14bd9.jsonl:74` |
| first CRF1 Add File | `/home/work/ai-graphics_33/.codex/sessions/2026/09/23/rollout-2026-09-23T22-01-58-01a0ce5b-ce8f-7e21-a7f4-182f9f9b0b6e.jsonl:48-50` |

### Appendix B. 비파괴 조사 명령 형태

아래는 이번 조사에서 사용한 read-only 검사 계열이다. 생성 command는 §6에서 인용만 했고 실행하지 않았다.

```bash
## Metadata, no encode:
ffprobe -v error -show_streams -show_format -of json <existing.mp4>

## Encoded packet PTS/count, no pixel decode:
ffprobe -v error -select_streams v:0 -show_packets \
  -show_entries packet=pts,dts,duration -of json <existing.mp4>

## Codec extradata / x265 SEI, no encode:
ffprobe -v error -select_streams v:0 -show_streams -show_data -of json <existing.mp4>

## SHA: Python hashlib reading bytes; no model import.
## History: git log --all / git diff / git blame / git status, GIT_OPTIONAL_LOCKS=0.
## Text: bounded keyword search; metadata/session lines parsed as data.
## Existing framemd5 files: payload hash columns compared as text only.
```

이 보고서의 “FOUND EXACTLY”는 남아 있는 실제 argv/sidecar/실행 로그와 bytes 연결에 대한 판단이다. 이번에 동일 영상을 새로 인코딩해 reproduction PASS를 확인했다는 뜻이 아니다.
