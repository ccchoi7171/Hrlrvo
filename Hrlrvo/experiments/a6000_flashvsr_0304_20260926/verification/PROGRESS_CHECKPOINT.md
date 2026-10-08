# Progress checkpoint — superseding update 2026-09-26 17:53 UTC

**User authorized A6000-only acceleration; DEPLOYED final-only parallel path, preserving all4 inference PIDs.**
Current authoritative details: `../optimizations/final_only_v2/DEPLOYMENT.md`. Earlier raw+serial postprocess ETA below is superseded.

- New remote post session `flashvsr_finalonly_0304_20260927`, windowsfinal_v04/final_v03, coordinatorPIDs1985689/1985690, waitingfororiginal2shards each. TakeoverPASS17:52:47UTC (9/27 02:52:47KST).
- Oldwaitingcoordinator1870245/1870250 retiredonlyafteridentity/children/state/artifactchecks andnewstandbybarrier. FourGPUPIDs1870485/1870504/1870324/1870332 unchanged,6/85tilescompleted each,nexttileadvancing;14,564MiB/GPU. SameGPUtmuxflashvsr_0304_20260926.
- Raw16K merge/encodeSKIPPED intentionally; finaldoesnotreadraw. Exactpixelmath threaded8/vid anddual finalcontinuouslibx265enables42%shortbenchmarkmakespangain. Samequalities/seeds/models/170geometry/1952timeline. NoH200productionchange.
- Newstatusstatus/final_only_v2/video03.json,video04.json; oldstatus/video*.jsonaretruthfulREPLACED_BY_FINAL_ONLY_V2 pointers. Logslogs/final_only_v2/video03.log,video04.log. Finaloutputpathsunchanged.
- Final118testsPASS,ruff+compilePASS,independentreviewAPPROVE0issues; actualremoteoriginal/threadedfullRGBhashsame,4frameactualx265serial54.667svsdual31.758s. Onlyshortfixture repeatedcommittedtiles—notfullproductionIO/GOPmeasurement.
- NewoverlaymanifestSHA5d5eda9c4076b90341011f827c94744758f67b0b19428cf6cd49c5a8c5224507; scripts/pinsinDEPLOYMENT.md andoptimization_manifest.json. Frozenoldassets/gatesstillPASS. DoNOTeditrunningpinnedcodecasually.
- ProvisionalKSTtileETA9/2716–19; finalboth9/2720:00–9/2801:00 (~4–6hnewpostpath estimate, lowconfidenceuntilrealpostrate). Old9/29–30ETAobsoletebecauseoldraw+serialpathremoved.
- H20002samePID2979965 chunkindex21 step6at17:52:34UTC;01watcher1939579alive;05DONEunchanged. NeverkillH200.
- Durablelocalcopiesevidenceverification/final_only_v2_takeover.json andfinal_only_v2_health.json;benchmarkdecisionunderverification/final_only_v2_bench/. NoactiveOMXworkflow; priorstatewriteambiguouslockleavealone,notepadcheckpointpossible.

## Historical superseding update 2026-09-26 16:57 UTC

**A6000 full jobs are now really running. All four first full 1952-frame tiles committed PASS.**
See `LAUNCH_AND_STATUS.md` beside this file for current PIDs, frozen pins, launch proofs, parameters, paths, H200 status, and conditional ETA. The older preparation notes below are historical and no longer current.

- Full remote tmux: `flashvsr_0304_20260926`, launched16:44:31UTC (9/27 01:44:31KST). Four GPU windows + two waiting postprocessing windows. No pending transfers or compatibility gates.
- Actual GPU PIDs 1870485/1870504/1870324/1870332 for physical GPUs0/1/2/3; 04 global0–84/85–169 and03 global0–84/85–169 respectively. Each ~13286MiB VRAM. All four initial full tiles PASS at16:55:52–16:56:20UTC.
- 6202 source files SHA PASS; exact102packageversions PASS; all4x29frame compatibility PASS16:44:21UTC. Current62testsPASS. Only isolated runtime native BSA binary rebuilt for Ampere from identical source.
- H20002 originalPID2979965 continuing;19/41 chunks929/1952frames completed16:57:01UTC; chunkindex19 started. Drainingbatch1228715 and01-only watcher1939579 alive.05 realDONE retained. No subsequent H200upscale.
- User latest request: confirm health and ETA. Evidence read only, no new16Kdecode/benchmark. GPU stage rough9/27 17–21KST. **Final both provisionally9/29–30, not GPU-stage time**: existing05 CPU rawmerge+encode16h10 and centercomposite7h45 pervideo; remote CPU postprocessing serialized; Gold5218R actualrate notyetknown. H20002~9/27 11:30KST;01~9/28 05–06KST ifsamepace.
- Remote proofs copied to local `verification/remote_launch_evidence/`. Health observation `health_observation_20260926T1657.json`. Do not edit frozen active code without repinning/review, do not stop production tmux on task close.

## Historical checkpoint 2026-09-26 15:14 UTC

User explicitly authorized complete migration + real full launch, and most recently reaffirmed identical seed/detail parameters. Status questions did not pause the task. Do not stop at preparation.

## H200 already safely split
- 05 naturally DONE 14:10:04 UTC. Preserve output; no further full decode.
- Original batch1228715 received one parent-only SIGINT; log proves no-new-jobs drain. NEVER signal again without new cause/identity audit.
- 02 Outpainting2979965 remains original PID, chunk_index14 step29 at15:14. Must never interrupt.
- Original04 pre-inference runner1807436 was retired after drain; historical04 FAILED_UPSCALE is intentional migration cancellation. Outpainting04/03 artifacts unchanged.
- 01-only continuation watcher1939579 in hrlrvo_train:outpaint_only_01_20260926, pane%452; waits02 PASS/parent exit/locks then run_cert.sh --stage outpaint --resume. Separate state verification/continuation_state.json. No H200 followup upscale.

## Isolated roots/auth
Local /home/work/ai-graphics_33/Hrlrvo/experiments/a6000_flashvsr_0304_20260926
Remote /home/oem/hrlrvo/experiments/a6000_flashvsr_0304_20260926
SSH oem@203.253.25.176 port2222, key $HOME/.ssh/codex_school_a6000_20260923_ed25519, knownhosts $HOME/.ssh/known_hosts_codex_school_a6000, IdentitiesOnly BatchMode StrictHostKeyChecking=yes. No password required; never print/store provided password.

## Active preparation/transfers (not full GPU launch yet)
- Remote dedicated tmux flashvsr_prepare_0304_20260926:prepare, pane%72, PID1832822, started15:13:56UTC.
- Runs tools/remote_prepare.sh with trusted manifest pins. Current WAIT environment archive/wheel. Durable sequence: SHA archive+wheel -> isolated env extraction/conda-unpack/no-deps Amperewheel install -> wait allsourceinputfiles -> allSHA verification -> environment preflight -> four29frame smoke -> gated6window launch-full.
- Remote logs/remote_prepare.log, verification/preparation_exit.txt on exit, status/*.json.
- Local env rsync unified session82037 -> logs/transfer_env.log (~945MB/3.8GB25%at15:14).
- Local sequential input rsync session8932 -> logs/transfer_inputs_parallel.log (04out~973MB/3.7GB26%; then03out,04anchor,03anchor).
- Small assets session22750 completed SMALL_ASSETS_PASS.
- FFmpeg exactH2006.1.1 bundle transferred/unpacked/tested version, session31447done.
- Amperewheel transfer session3971done; codebundle session51008done.
- Remote direct maincheckpoint download PID1816808, logs/download_main_weight.log. ~60%at15:05, likelydone~15:30. Original staged wrongsubdir FlashVSR-v1.1; remote_prepare safely SHA+mv to checkpoint/ once PASSmarker andcheckpointdir exist.
- Env/input transfers are bottleneck, expected another~1.5h ifrateimproves afterweightdownload. Do not inventETA.

## Reviewed/frozen code
- tools/remote_control.py orchestrates exactasset/model/sourcecode/complete102packageversions/envfileidentities, actualGPUUUIDs, four29frame smoke, tokenized start barrier for6tmuxwindows, four supervisors,2postcoordinators.
- tools/tile_worker.py fixed256/32,bf16,seed71634910+globalindex, sparse2/local11, ffthreads32/libx265, full1952 causal same original neuralcode. GPU0/1=04tiles0-84/85-169; GPU2/3=03same.
- tools/merge_shards.py exact170globaltiles+geometry; raw target MUST be under runs/<video>/raw_flashvsr (fixed prior containmentbug).
- tools/postprocess_video.py + package/src/cert_pipeline/flash_composite.py reuse production pixel math; fresh-only raw/final/report, no duplicate16Kdecode.
- production upscale.py unchanged SHA3db9c0524418070a87b26b948612f8988cd47f007c6e7739a89d3911ed0d703d; productionmedia_utils unchanged314450...b8a9.
- 60testsPASS,ruff/compile/bash-nPASS; code-reviewerAPPROVE, no remainingcodeblocker. ActualGPUtests stillpending.
- Code manifest SHA09978959cda6152cd84753b67ac3ecb68ab305d4b339b6826628b7406ce46845
- Runtimeffmpegmanifest SHA5ef96a13c4168319933098cbb0e2e5b8ddff0e357c52e09271cf8b69f4d2f17a
- remote_prepare.sh SHA108653992aa75ef87f49cb0d24d581990ce472b56294ab4f247bd622c1e15e74
- remote_control.py SHA59fae0af18e5fb5a98f30c6bacd7eaf12cd5a3c52064c40176fc780e49bee33d
- codebundle SHA65251c0b5bad2e2f866dcc89bd1c43babcd65f484d47d78ecd318e0d48df46a8 verifiedremote beforelaunchprep.
Do not modify frozen code unless needed; repin/redeploy/reverify if changed.

## Ampere attention compatibility
Original copiedenv BSA binary has onlysm90, unsuitable A6000. Bounded local build from exactpinnedsource5701/5701 unchanged finishedexit0. Wheel62,531,199bytes SHA7f617e49b8c65879585a0be0ae8df93ff43e9a4ad75be2c09b8afb6a3329164d. ExtensionSHA c20a5184e37d2ff6ec648c465b396e8813d01ef126b04a0722b2575e3dd6a231.
24sm80cubins+24compute80PTX, GLIBC<=2.32 (remote2.35), GLIBCXX<=3.4.32 (copiedenv3.4.34). CUDA12.8 compile vs torch2.6cu124 standardminorwarning; actualGPUsmoke mandatory. No H200envinstallation or GPUoperations. Proofverification/bsa_ampere_build.json. RuntimeLD_LIBRARY_PATH set isolatedenv/lib; controllerinvocations-I andenv-uPYTHONPATH-uLD_PRELOAD.

## Next
1 Monitor transfers/preparation with 10-15min pauses; keep02progress/watcher checks bounded. Remote preparation now automatically performs safe gates and launch oncefilesarrive. Iferror, inspectlogsandfix onlyisolatedR, neverH200.
2 When realGPUtests/launch run, inspect compatibility_test.json / source_transfer_verification.json / environment_preflight.json and status/launch.json.
3 Verify allfourfullworker PIDs/tmux windows and actualfirst1952frame tileprogress, initialVRAM/speed, assignments. Fullsession flashvsr_0304_20260926 windows gpu0_v04/gpu1_v04/gpu2_v03/gpu3_v03/post_v04/post_v03.
4 Report H200state + remoteactualstartdetails UTC/KST and paths. Do not claim fullvideoscomplete; user asks actualrunstart.

## Update 15:37UTC (supersedes preparer PID / code pins above)
- Latest user also asks to report final A6000 tmux session/window names afteractualstart. Acknowledged; do not omit.
- Main HF checkpointdownload completed and SHA PASS (~15:29). Remotehelper willmove it intocheckpoint/ afterenvinstall. Allsmallassets,ffmpegbundle,Amperewheel,codebundle done.
- Env+04input transfers now~48%/51%, each~1.1MB/s aftercheckpointdownloadreleasedbandwidth. Envlikely~16:05UTC; remaining03/anchorsfollow04. Keepwait, don'tclaimfullstart.
- Proactive discovered sourceenv diffsynth1.1.7 is editable externalmetadata. Archiveinspection proved bothpacked diffsynth.egg-link/easy-install.pth still point at H200 externalrepo. Packagedbyte-identical repository hassameegg-info, so added exactprefixatomicrelocation inremote_prepare afterwheelinstall (onlyisolatedmetadata, noinferenceparams/sourcechanges), actual tempdir regression.
- Safely stopped ONLY our ownwaitingremotepreparerPID1832822 afteridentityandenv-absentchecks. DidnotstopH200,transfers,orGPUtasks. Its exitrecordretained asverification/preparation_exit.before_editable_relocation.txt. Refroze/redeployedbundle andrespawned SAME ownedpreppane%72.
- NEW prep PID1840874, started2026-09-26T15:36:27UTC, tmuxflashvsr_prepare_0304_20260926:prepare, waitingpackedenv.
- NEW code_manifest SHA5ad0a4e807901b23337ffba9ce049b417e81260e8c809ab4d0fa6d6abc5fbe87
- NEW remote_prepare.sh SHA85c67b0f29fe2a8a42a8ffb9735d37c17274ca4a08e2a4ca3307e9652dd1de8d
- NEW codebundle SHA11feb21f52615bf5e400a25fcda443515f3ff2657b3c6a1e42eca7a230f6c3d4
- remote_control andruntimeffmpegmanifest SHAunchanged fromabove.
- 61testsPASS; ruff/bash-n/compilePASS; reviewer approvedrelocation and suggestedisolationprefix, whichwasfixed beforefreeze.
- Sourceenvhas102distributionversions; BSAnewwheel Name/version matchesoldexactly. Relocationmakesdiffsynthvisibletoimportlib.metadata and actualpackagerepository. Needactualenv/GPUteststill.
- H20002samePID2979965active15:36;01watcher1939579alive.

Checkpoint note: OMX state write returned canonical lock owner published-binding-mismatch; no lock or runtime state was manually changed. This Markdown and LAUNCH_AND_STATUS.md are the durable checkpoint; notepad write succeeded.
