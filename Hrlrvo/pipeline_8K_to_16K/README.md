# 8K → 16K certification workspace

> **현재: READY FOR FULL QUEUE — 코드/설정/CPU 및 기존 C0 evidence 기준.** 05 C0 latent·native RGB·decoded native24fps RGB가 모두 exact 일치했습니다. C1+ rebasing/hybrid carry/strength1.0/직접4K LineArt→30×17 Global prior 연결은 CPU·AST·mock 테스트로 확인했습니다. 113f GPU smoke는 사용자 요청으로 C0 도중 중단했으며 **C1 결과 품질·Full/Flash runtime 검증 PASS를 의미하지 않습니다.** 이후 추가 추론 없이 확인했습니다. [최종 통합/queue audit](docs/RECURRENCE_LINEART_INTEGRATION.md).

> **이전 Stage 0001 작업: PREPROCESS REPRODUCED.** Historical CRF18 전체 command contract로 수정한 공통 Stage 0001을 01~05에 실제 CPU 실행했습니다. **각 영상 첫 65/65 및 전체 1,952/1,952 decoded RGB frames가 일치**했고, MP4 SHA256도 5개 모두 canonical과 같습니다. [실측 결과](docs/PREPROCESS_REPRODUCTION_01_05.md). 이 Stage 0001 수정 작업 당시에는 GPU Outpainting seed 재현성이 별도 미검증이었고 Outpainting/FlashVSR/CUDA를 실행하지 않았습니다. 이후 recurrence 작업의 C0 검증 결과는 위 최신 audit에 별도로 기록합니다.

The three original directories (`pipeline`, `VideoOutpainting`, `Upscaler`) remain read-only. Migration originally changed orchestration/layout only. The later explicitly requested recurrence integration changes C1+ behavior while preserving C0, checkpoints, preprocessing and FlashVSR. All LongRange/chunk artifacts remain24fps; only the complete merged native master is losslessly retimed once to16fps for FlashVSR.

**Previous certification draft: CRF1. Corrected historical reproduction contract: CRF18.** The earlier CRF1 requirement did not reproduce the seed-search input. The user's corrected requirement restores the complete historical command, including decoder threads and fixed media validation, not CRF alone. The old CRF1 attempt's 05 result (0/65) remains preserved in `docs/REPRODUCIBILITY_05.md`; it is not the result of the corrected producer. Preprocessing PASS does not establish Outpainting numerical reproduction.

## Commands

```bash
./run_cert.sh --config configs/certification/05.json
./run_cert.sh --config configs/quick/05.json
./run_cert.sh --config configs/certification/05.json --stage preprocess
./run_cert.sh --config configs/certification/05.json --stage outpaint
./run_cert.sh --config configs/certification/05.json --stage upscale
./run_batch.sh --config configs/batch/certification.json
```

**Upscale output mode:** Fresh Full/Quick/Batch runs default to `final_only`:
FlashVSR tile inference, direct final 16K center composite and original audio,
without the redundant raw 16K MP4 encode. To retain the historical raw MP4,
use `./run_cert.sh --config configs/legacy_raw/05.json --stage upscale` after
canonical Outpainting PASS. Existing completed legacy results remain reusable
with `--resume`; they are not rewritten. See
[Upscale output modes](docs/UPSCALE_OUTPUT_MODES.md).

CLI inference conditions are not overrideable: JSON controls seed/frames/steps/model/prompt. `--stage` selects orchestration only. `--resume` validates existing manifests; `--force` archives only the selected stage's owned output subdirectories. Existing references are never touched. Select the correct config and inspect output/free disk first.

## Fresh Full versus Stage/cache

- Fresh Full begins at 8K and invokes the SAME `run_preprocess_stage`, `run_outpaint_stage`, `run_upscale_stage` used by stage-only. It does not silently skip copied PreProc. Imported canonical MP4/JSON files remain immutable. Fresh results are published to `data/01_PreProc_2K_1920x1080/metadata/<ID>/generated/<ID>_PreProc_2K.mp4`, with a producer receipt; prior generated results are preserved before replacement.
- `--stage outpaint` never calls preprocessing. The shared selector uses the generated artifact if present, otherwise the canonical cache, and verifies SHA, source provenance, frame count and historical producer contract. Generated receipts additionally require matching implementation/toolchain identity. Invalid generated artifacts fail closed. `resolved_config.json.resolved_preprocess_artifact` records the actual input path.
- `--stage upscale` requires a committed outpaint receipt and SHA + native manifests; it does not infer outpaint.
- Batch uses the same validated PreProc selection; the producer immediately starts the next Outpaint independently of the FIFO Upscale consumer. It does not automatically preprocess missing inputs; use the shared `--stage preprocess` beforehand if needed. Child OS RNGs are isolated. 1 Outpaint + 1 Flash may overlap; no two of either. Queue implementation is unchanged, and normal queue inference was NOT run in this correction task.
- Quick retains65fOutpaint and selects21f forFlash. Outputs live beneath eachID's `quick/`, separate from Certification-owned directories.

## Environment

Keep the existing `Hrlrvo` and `Hrlrvo-FlashVSR` environments. Shell selection is copied from the audited entrypoint. Optional `CERT_OUTPAINT_PYTHON` / `CERT_FLASH_PYTHON` select installed interpreters, not inference parameters. Do not reset GPUs or stop other users' work. Flash retains auto tile behavior. Dependencies are recorded in copied `src/manifests/`.

## Evidence

- `docs/PREPROCESS_REPRODUCTION_01_05.md`: actual five-video reproduction, hashes, command differences and CPU acceptance evidence.
- `docs/PIPELINE_EXECUTION_AUDIT.md`: current Stage 0001 addendum plus clearly archived original audit; old directory/log/line references describe the earlier snapshot.
- `docs/MIGRATION_DIFF_AUDIT.md`: current correction classification plus preserved migration evidence.
- `docs/HISTORICAL_PREPROCESS_PROVENANCE.md`: historical producer/source-version evidence and current correction addendum.
- `docs/REPRODUCIBILITY_05.md`: preserved previous CRF1 failure and the then-unexecuted GPU gate; this historical report is not the newer recurrence verdict.
- `docs/RECURRENCE_LINEART_INTEGRATION.md`: current C0 exact evidence, C1+ code/CPU acceptance, cancelled smoke record, native24/final16 boundary and actual Full FIFO configuration.
- `docs/DIRECTORY_LAYOUT.md`, `docs/BATCH_QUEUE.md`.
- `state/migration/`: preserved original copy/migration evidence. `state/preprocess_reproduction/`: before snapshots, current diffs, CPU tests and scope checks. `data/validation/preprocess_reproduction/<ID>/`: new isolated videos, full RGB frame hashes and comparisons. Canonical references are not overwritten.

Disk warning: Full Flash persists tile_count×N PNGs. The default final-only
path uses bounded float32 row-band assembly for one final composite encode;
the opt-in legacy path additionally stores and encodes a raw 16K MP4. GPU
inference and full 16K execution require real validation; static/CPU tests
alone do not prove runtime success.
