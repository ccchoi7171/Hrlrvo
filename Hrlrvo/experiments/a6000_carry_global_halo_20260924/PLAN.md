# A6000 four-arm execution

Approved: implement and launch four isolated FHD runs, no overwrite.
Reference: longrange_fhd_recurrence_test_20260923 strength_1p0 patched source.
Common: strength=1.0, temporal rebase=true, checkpoint 7500, seed=71634910+absolute_start, 65/17/48, 30 steps, same input/prompts/CFG/STG. All arms fresh C0; historical reference reused C0 before C1 resume, recorded as a reproducibility limitation.
01 exact baseline; 02 no carry with baseline authoritative anchor rule retained; 03 fixed Global contribution scale=1.4 steps0..9 then1.0; 04 Native core-Q/output with clamped KV halo radius1 canonical Native spatial token (32px stride; user explicitly forbids radius2). Halo added anchor read support is never written outside original Q core.
Only isolated arm source edited. Original model files and input shared read-only.
Checks before launch: CPU targeted regressions; source diff whitelist; checkpoint/input content verification; effective config equality except declared intervention; remote compile/import; idle GPU check.
Remote work via tmux hrlrvo. Per-arm code/config/logs/outputs/chunks/latent_states. Finish task on four healthy inference processes; report results as pending, not validated quality.
