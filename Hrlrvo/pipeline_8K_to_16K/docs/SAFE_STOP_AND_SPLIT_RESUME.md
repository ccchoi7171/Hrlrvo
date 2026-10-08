# Full Batch 안전 중단 및 Outpainting / Upscale 분리 재개 방안

작성일: 2026-09-24

작업 root: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K`

> **이 문서는 향후 사용자가 전환을 요청했을 때 적용할 운영 방안이다. 지금 중단하거나 실행 모드를 바꾸라는 지시가 아니다.**
> 문서 작성 과정에서는 source/config를 수정하지 않았으며, 실행 중인 batch/child에 어떤 signal도 보내지 않았다. 새 inference, model load, FFmpeg 처리, 중단·재개 실험도 하지 않았다.

## 1. 결론과 증거 수준

**완료 artifact를 보존하고 Outpainting과 Upscale을 나누어 실행할 수 있는 구조다.** 다만 현재 batch에는 실시간 `upscale=OFF` 또는 `stop-after-chunk` 인터페이스가 없다.

| 구분 | 의미 | 확인 수준 |
|---|---|---|
| 영상의 Outpainting stage 전체 완료 후 분리 | 완성된 4K/16fps와 receipt를 보존하고 Upscale은 나중에 실행 | CONFIRMED BY CODE; 권장 경계 |
| 완료 chunk를 이용한 Outpainting resume | 연속된 완료 chunk를 검증하고 prefix/RNG/hybrid carry 복원 후 다음 chunk 생성 | CONFIRMED BY CODE |
| denoising 중간 step에서 재개 | 예: 13/30에서 14/30으로 바로 이어가기 | 지원하지 않음 |
| batch parent만 drain 요청 | 현재 child는 완료시키고 후속 job 실행을 막는 handler가 있음 | CONFIRMED BY CODE; 실제 signal 전환은 NOT RUNTIME-VERIFIED |
| 아무 시점 강제 종료의 무손실 보장 | 저장 중단·파일 손상·미완료 chunk까지 모두 보존 | 보장하지 않음 |

현재 Full 실행에서 실제 중단→재개에 따른 numerical equivalence를 시험한 것은 아니다. 기존 C0 exact-preservation PASS는 recurrence 변경 전후 C0에 대한 증거이며, 모든 interruption 시점의 resume equivalence 증거로 확대하지 않는다.

## 2. 변경하지 않을 것

- 기존 `pipeline`, `VideoOutpainting`, `Upscaler`는 READ ONLY reference다.
- 모델·checkpoint·seed·prompt·30 steps·65/17/48 LongRange·recurrence·LineArt·PreProc은 바꾸지 않는다.
- job/config bytes와 artifact 위치를 유지한다. batch resume은 batch/job JSON SHA에 결합되어 있다.
- `--force`, 기존 artifact 삭제/이동, state JSON 수동 조작으로 오류를 우회하지 않는다.
- tmux 창 삭제, 광범위한 process kill, process-group signal, GPU reset, 다른 작업 종료를 하지 않는다.
- `.lock` 파일이 존재한다는 이유만으로 삭제하지 않는다. 보호 장치는 파일 존재가 아니라 실제 `flock` 소유권이다.

## 3. 현재 인터페이스와 한계

### 동일한 stage 구현

```text
Full run_cert(all) ────────────────┐
run_cert(--stage outpaint) ────────┼→ run_outpaint_stage() → run_outpaint() → sample_long_range()
run_batch producer → run_cert ────┘

run_cert(--stage upscale) → run_upscale_stage()
```

근거: `run_cert.py:1015–1071,1105–1144`, `run_batch.py:232–255,320–383`.

`--stage outpaint`는 검증된 기존 PreProc을 읽으며 preprocessing이나 FlashVSR을 실행하지 않는다. `--stage upscale`는 완료된 Outpainting deliverable을 검증한 뒤 실행한다.

### live toggle은 없음

- `run_batch.py:154–160`: Outpaint/Upscale worker 모두 정확히 1이어야 한다.
- `run_batch.py:278–308`: batch 설정·resume 상태는 시작할 때 읽는다. live control file을 polling하지 않는다.
- `run_batch.py:340–369`: Outpainting 성공 시 자동 enqueue → Upscale 호출.
- `run_batch.py:414–418`: batch CLI는 `--config`, `--resume`만 제공한다.
- `run_batch.py:213–228`: 설정 bytes를 바꾼 후 기존 batch `--resume`을 하면 request mismatch로 거부된다.

따라서 JSON에서 `upscale_workers=0`으로 바꾸거나 실행 중인 `.py`를 수정하는 방식으로 전환하지 않는다. Upscale child 하나만 끝내도 consumer는 다음 대기 영상을 계속 처리하므로 전체 Upscale 비활성화가 아니다.

## 4. 나중에 사용자에게서 받을 요청 예시

> 현재 batch를 안전하게 분리해줘. 가능하면 지금 실행 중인 영상의 Outpainting stage는 끝까지 완료시키고, 새로운 Upscale은 시작하지 않도록 해줘. 기존 artifact를 보존한 뒤 Outpainting만 05→04→03→02→01 순서로 이어가고, Upscale은 나중에 별도 요청할 때 진행해줘. 먼저 현재 process와 commit 경계를 재확인해줘.

만약 더 빨리 멈춰야 한다면:

> 현재 영상 완료까지 기다리지 말고, 저장 완료 chunk를 최대한 보존하는 중단·재개 방식으로 전환해줘. 진행 중 chunk 재계산 가능성은 기록해줘.

두 요청은 다르다. 첫 번째는 현재 **영상 전체** 완료를 기다리는 방식이고, 두 번째는 **chunk 경계와 미완료 작업 손실**을 고려하는 방식이다. 이 문서 자체는 어느 방식도 실행하도록 승인하지 않는다.

## 5. 전환 직전 필수 재확인 — READ ONLY

향후 요청 시 agent가 다음을 직접 확인한다. 과거 PID나 과거 chunk 번호를 재사용하지 않는다.

1. `hrlrvo_train`의 실제 window/pane 및 해당 shell·batch·Outpaint·Upscale·FFmpeg process tree.
2. PID뿐 아니라 command line, 시작 시간, PPID, PGID를 확인하여 대상 run을 식별.
3. `state/batch_request.json`, `state/batch_state.json`, `logs/batch_pipeline.log`.
4. 현재 `OUTPAINT_CHUNK_DONE` / `inference_chunk_complete`와 native commit 기록.
5. 현재 Upscale 실행 여부. 실행 중이면 그 stage도 완료시킬지, tile 단위 재개를 허용할지 요청 범위에 맞춰 결정.
6. config·input·runtime identity, disk free space, stage lock 소유 process.
7. 전환 요청 시각, 선택한 방식, 마지막 완료 chunk 및 현재 미완료 chunk를 별도 전환 기록에 남김.

문서 작성 시점(2026-09-24 약 09:57 UTC)의 관찰은 05 `OUTPAINT_RUNNING`, 04~01 `PENDING`, 완료 chunk 22였다. 이는 **과거 스냅샷**이며 나중에 사용할 중단 지점이 아니다.

## 6. 권장 방안 A — 현재 stage를 끝까지 완료시키는 drain

### 코드가 제공하는 동작

`run_batch.py:395–400`은 parent의 `KeyboardInterrupt`를 받으면:

```text
stop_requested.set()
[QUEUE] interrupt_received action=finish_running_children_no_new_jobs
producer / consumer thread 종료 대기
```

를 수행한다. producer는 다음 video 시작 전에, consumer는 다음 job 실행 전에 해당 event를 확인한다(`:323–324,:360–361`). 이 handler 자체는 child를 종료시키는 코드를 호출하지 않는다.

### 운영상 의미

- **현재 실행 중인 child stage를 완료시킨 후 새 job을 시작하지 않는 방식**이 우선 후보다.
- 이를 사용한다면, 미래 실행 시 process tree를 확인한 뒤 **batch parent 하나에만** 제어를 전달해야 한다.
- **tmux에서 일반 Ctrl+C를 누르는 것과 다르다.** 현재 parent와 Outpaint child는 같은 foreground process group에 있었다. 터미널 Ctrl+C는 child에도 도달할 수 있다.
- 사고를 막기 위해 이 문서에는 과거 PID를 넣은 signal 명령을 제공하지 않는다.
- 이 방식은 현재 영상을 다음 chunk에서 멈추는 기능이 아니다. 남은 chunk 전체·merge·retime·stage 검증까지 기다릴 수 있다.

### 반드시 확인할 종료 조건

1. drain 로그가 실제로 남았는지 확인.
2. 이미 실행 중인 child는 계속 정상 진행하는지 확인.
3. signal과 worker dispatch가 동시에 일어날 수 있으므로, 직후 process tree를 다시 확인. 경합으로 이미 시작된 작업이 있다면 없는 것으로 가정하지 않는다.
4. 현재 child의 `OUTPAINT_DONE ... status=PASS`, `stage_complete.json`, final deliverable을 확인.
5. 다음 Upscale/Outpaint가 새로 시작하지 않았는지 확인.
6. **parent 종료나 `batch_summary`만으로 전체 종료를 판단하지 않는다.** child/grandchild가 실제로 모두 종료했고 stage lock이 해제되었는지 별도로 확인.
7. batch 종료 코드가 1이어도 `interrupted=True` 때문일 수 있다(`run_batch.py:48–54,438`). 개별 stage 실패와 구분한다.

**NOT RUNTIME-VERIFIED:** 이 drain 경로는 이번 문서 작업에서 실제 signal로 검증하지 않았다. 로그/종료 상태가 위 의도와 다르면 재차 signal을 반복하거나 child를 강제 종료하지 말고 상태를 보존하여 진단한다. “parent에게 signal 하나만 보내면 무조건 안전하게 끝난다”는 보장은 하지 않는다.

## 7. 대안 B — 영상 도중 중단 후 chunk resume

빠른 중단이 필요할 때의 대안이며, A보다 신중한 artifact 검증이 필요하다.

- sampler에는 `stop-after-chunk` 제어 인터페이스가 없다. chunk 완료 로그를 읽고 중단을 결정하는 사이에도 다음 chunk가 시작할 수 있다.
- 가능한 기준은 단순 denoise `30/30`이 아니라 **native commit + certification artifact 저장 완료**다.
- interruption에 따라 현재 chunk의 denoising, decode 또는 audit encoding을 다시 수행할 수 있다.
- GPU kernel 실행 중 interruption 반영까지 시간이 걸릴 수 있다. 즉시 종료되지 않는다고 무차별 강제 종료하지 않는다.
- batch parent를 먼저 어떻게 제어할지도 함께 결정해야 한다. Outpaint child만 중단하면 producer는 실패로 기록하고 다음 영상을 실행할 수 있다(`run_batch.py:332–339`).

예: 완료 chunk 22가 유효하고 23번째 denoising 도중 종료되었다면, resume은 22개를 검증·재사용하고 23번째를 다시 생성하는 구조다. 중간 step의 latent snapshot으로 이어가는 기능은 없다.

## 8. 완료 chunk의 보존·재개 원리

### 실제 commit 순서

```text
chunk final generated latent 기록
→ hybrid causal latent carry 저장
→ finalized RGB segment 저장
→ next 17-frame FP32 RGB prefix + CPU/CUDA RNG 저장
→ native chunk_<index>.json atomic commit (완료 표시)
→ certification MP4 / carry archive / artifact receipt 기록
→ OUTPAINT_CHUNK_DONE
→ 다음 chunk
```

근거: `inference.py:2258–2262,2272–2335`, `streaming_video.py:420–538`, `outpaint.py:350–450,808–851`.

`previous_prefix_0.pt` / `previous_prefix_1.pt` 교대 저장은 미완료 새 chunk가 **직전 완료 chunk**의 prefix를 덮어쓰지 않게 한다(`streaming_video.py:415–418`). 모든 과거 prefix가 두 파일에 남는다는 의미는 아니다. 과거 기록용 사본은 각 chunk의 `carry_prefix_state.pt`에 별도로 보존한다.

### 나중에 보존 상태를 확인할 파일

```text
data/02_Outpaint_4K_3840x2160/<ID>/
├── chunks/native/
│   ├── run.json
│   ├── model_ready.json
│   ├── chunk_XXXXXX.json
│   ├── segment_XXXXXX.npy
│   ├── previous_prefix_0.pt
│   ├── previous_prefix_1.pt
│   └── causal_latent_carry_XXXXXX.pt
├── chunks/latents/chunk_XXXXXX/
│   ├── generated_latent.pt
│   ├── metadata.json
│   ├── carry_prefix_state.pt
│   ├── causal_latent_carry.pt
│   └── certification_artifacts.json
├── chunks/rgb/chunk_....mp4
├── lineart/
│   ├── source_frame_4k.png
│   ├── lineart_4k.png
│   ├── global_line_density.npy
│   ├── global_line_strength.npy
│   └── lineart_metadata.json
└── metadata/
    ├── stage_request.json
    └── input_identity.json
```

### resume의 실제 검증

- source/config/frames/seed/recurrence implementation identity 비교.
- native 완료 기록이 chunk 0부터 연속인지 검사.
- segment shape/dtype/size/pixel SHA 확인.
- latest FP32 RGB prefix/RNG size/SHA/shape/dtype 확인 및 RNG 복원.
- hybrid latent carry SHA/shape/dtype/finite 확인 및 복원.
- 해당 영상의 C0 LineArt reference 재검증.
- 검증된 완료 chunk는 generation loop에서 skip.

근거: `run_cert.py:773–797,1015–1049`, `streaming_video.py:372–381,541–674`, `inference.py:2013–2062,2108–2123,2180–2181`, `recurrence_artifacts.py:41–59`.

모델은 필요한 경우 다시 load하지만, 이는 완료 chunk의 denoising을 다시 수행한다는 뜻은 아니다. resume validation의 SHA/영상 검사 자체는 시간이 걸릴 수 있다.

### 저장 도중 interruption의 한계

- temp write → flush/fsync → atomic replace를 사용하지만, chunk 전체의 모든 파일이 하나의 filesystem transaction인 것은 아니다.
- native commit 직후 audit MP4/archive 생성 도중 중단되면 wrapper가 가능한 artifact를 복구한다(`outpaint.py:752–782`).
- 생성 latent 누락/손상, 필요한 prefix 사본 소실, SHA mismatch 등은 자동으로 무시하지 않고 거부한다.
- `.partial`/`.tmp`나 오래된 manifest를 임의 삭제하여 PASS로 만들지 않는다. 중단 시점과 실제 commit을 먼저 대조한다.

## 9. 분리 재개 명령 — 향후 실행 예시, 지금 실행하지 않음

전제: 기존 batch와 해당 model child가 모두 종료했고, config/model/input/artifact가 유지되며, resume 검증이 통과할 수 있는 상태다. 기존 배치를 켜둔 채 아래 명령을 별도 창에서 경쟁 실행하지 않는다.

### 환경

```bash
cd /home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K
export CERT_OUTPAINT_PYTHON=/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python
export CERT_FLASH_PYTHON=/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-FlashVSR/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

환경 경로는 현재 run에서 사용한 값이다. 나중에 실행할 때도 존재 및 동일 환경 여부를 확인한다. stage dispatcher는 `run_cert.sh:5–14`를 사용한다.

### 05 Outpainting이 미완료라면

```bash
./run_cert.sh --config configs/certification/05.json --stage outpaint --resume
```

이미 해당 영상의 Outpainting final과 PASS receipt가 검증되었다면 이 명령은 생략하고 다음 영상으로 간다. 완료된 영상에 다시 outpaint resume을 호출하면 GPU generation은 skip되더라도 validation/merge/encoding 작업이 다시 발생할 수 있다.

### 이후 Outpainting만 04 → 03 → 02 → 01

아래는 05를 끝낸 뒤 나머지 네 영상이 아직 완료되지 않은 경우의 예다. 실행 전에 완료 목록에 맞게 대상 목록을 선택한다.

```bash
for id in 04 03 02 01; do
  ./run_cert.sh --config "configs/certification/${id}.json" --stage outpaint --resume || break
done
```

이 loop는 기존 stage 호출을 직렬 연결한 운영 예시이지 새 모델 구현이 아니다. 실패 시 다음 영상으로 넘어가지 않도록 `break`한다. 기존 batch의 실패 후 계속 정책과는 다르다. 실패 여부는 마지막 shell exit code만 보지 말고 stage 로그와 PASS receipt로 판정한다.

`--resume`은 기존 기록이 있으면 검증 후 이어가고, native session이 아직 없는 깨끗한 stage는 fresh generation으로 진입한다(`outpaint.py:752–759`). 잔여 파일은 있는데 session이 없는 상태는 자동 fresh로 덮어쓰지 않는다.

### 나중에 Upscale만 05 → 04 → 03 → 02 → 01

사용자가 별도 실행을 요청하고 각 Outpainting 완료 gate가 통과한 뒤:

```bash
for id in 05 04 03 02 01; do
  ./run_cert.sh --config "configs/certification/${id}.json" --stage upscale --resume || break
done
```

각 영상은 다음을 검사해야 FlashVSR로 넘어간다(`run_cert.py:1074–1102`).

- `metadata/stage_complete.json`: PASS 및 request 일치.
- `final/<ID>_Outpaint_4K.mp4`: 기록된 SHA 일치.
- chunk plan/results/lineage/inference manifest 존재.
- 3840×2160, 16fps, 요청 frame count.

native 24fps master나 개별 chunk를 FlashVSR 입력으로 수동 대체하지 않는다. LongRange는 계속 24fps이며, 전체 merge 후 한 번 retime된 final 16fps deliverable만 사용한다.

## 10. 이미 Upscale이 시작된 뒤 전환을 요청한다면

우선 가능한 경우 현재 Upscale stage도 정상 완료시키는 drain을 택한다. 더 빠른 중단이 요구되면 다음 제한을 명확히 기록한다.

- FlashVSR 재개 단위는 **완료된 spatial tile**이다.
- 해당 tile의 전체 temporal frame PNG와 emission metadata가 commit되고 검증되어야 재사용된다.
- 미완료 causal stream의 emission/프레임 하나에서 바로 이어가는 재개 기능은 없다.
- tile geometry, frame count, seed, request SHA, PNG size/SHA가 맞아야 한다. auto tile 계획이 바뀌면 기존 완료 tile 재사용을 보장하지 않는다.
- raw assembly/final encode 도중 중단되면 tile 재사용과 최종 encode 재수행을 구분한다. 임시 MP4를 완성 결과로 취급하지 않는다.

근거: `src/flashvsr/architect/FlashVSR_v1.1/upscale.py:747–806,1133–1164`.

현재 활성 tile을 강제 종료해도 작업 손실이 전혀 없다고 말하지 않는다. 이 문서 작업에서는 FlashVSR interruption/restart를 실험하지 않았다.

## 11. batch state와 stage receipt를 혼동하지 않기

- stage-only 실행은 `state/batch_state.json`을 갱신하는 batch worker가 아니다.
- 분리 실행 후 batch JSON이 과거 `OUTPAINT_RUNNING` / `UPSCALE_WAITING` 상태여도, 실제 완료 여부는 stage receipt/manifest로 확인한다. state를 손으로 `DONE`으로 고치지 않는다.
- 향후 **병렬 full queue로 복귀**하겠다는 요청이라면 기존 config 그대로 다음 명령을 사용할 수 있다.

```bash
./run_batch.sh --config configs/batch/certification.json --resume
```

이 명령은 **Outpainting-only가 아니다. Upscale consumer도 다시 실행한다.**

`run_batch.py:260–274`는 중단된 Outpaint 상태를 PENDING, 완료 Outpaint/이전 DONE 등을 UPSCALE_WAITING으로 정규화하고 stage resume 검증을 맡긴다. stage-only에서 끝난 작업이 queue JSON에는 반영되지 않았으면 추가 stage 검증/merge가 발생할 수 있다. config bytes를 바꾸거나 상태를 조작하여 그 비용을 임의 우회하지 않는다.

## 12. 다음 요청 시 agent가 확인할 체크리스트

- [ ] 사용자가 실제 중단/전환을 요청했는가? 단순 질문/문서 요청이면 실행하지 않음.
- [ ] 현재 process tree, 시작 시각, cmdline, PGID를 재확인했는가?
- [ ] 현재 stage 전체 완료를 기다릴지, 미완료 chunk/tile 재계산을 허용할지 명확한가?
- [ ] parent 제어와 child 종료를 구분했는가? tmux Ctrl+C를 안전 drain과 동일시하지 않는가?
- [ ] 신규 dispatch가 중지되었는지 실제 로그/process로 확인했는가?
- [ ] 완료 commit/receipt 및 필요한 prefix/RNG/carry/LineArt가 보존되었는가?
- [ ] 기존 batch뿐 아니라 모든 해당 child/grandchild의 종료와 lock 해제를 확인했는가?
- [ ] config/input/model/runtime identity가 유지되는가?
- [ ] 충분한 output disk free space가 있는가?
- [ ] Outpainting stage-only는 동일 구현이며 FlashVSR을 실행하지 않는가?
- [ ] Upscale은 검증된 final 16fps 4K만 입력받는가?
- [ ] queue state와 stage-only 완료 기록의 차이를 보고했는가?
- [ ] 재개 대상/마지막 완료 chunk/재계산 대상/미검증 위험을 기록했는가?

## 13. 보고 예시

```text
TRANSITION: DRAINED / INTERRUPTED_FOR_RESUME / NOT_COMPLETED
Original batch: EXITED / STILL_RUNNING
Stage children: EXITED / STILL_RUNNING
Last committed Outpaint chunk: <actual>
Uncommitted work to recompute: <actual or UNKNOWN>
Existing artifacts: PRESERVED / VALIDATION_FAILED
Next mode: OUTPAINT_ONLY / UPSCALE_ONLY / PAUSED
Runtime/model/config changed: NO
Resume verification: PASS / NOT_YET_RUN / FAIL
```

**최종 원칙:** 완료된 계산은 manifest 검증 후 재사용하고, 미완료 계산은 필요하면 재실행한다. 안전한 경계와 실제 process 종료를 확인하기 전에는 분리 runner를 시작하지 않는다. 현재 batch는 사용자의 미래 전환 요청 전까지 원래 목표대로 계속 실행한다.
