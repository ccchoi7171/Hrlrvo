"""Adapt the archived runner's orchestration, retaining its generation/commit loop."""
from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parents[1]
s=(root/'shared/reference_runner.py').read_text()
def replace(a,b):
 global s
 assert a in s,a
 s=s.replace(a,b,1)
replace("ROOT=Path(__file__).resolve().parents[4]\nCOPY_SRC=ROOT/'source/VideoOutpainting_test/main/src'\nANCHOR=ROOT/'input/05_anchor_960x540_24fps_1169f.mkv'\nBASELINE=ROOT/'configs/baseline_05_run.readonly_copy.json'", "ROOT=Path(__file__).resolve().parent\nARM=ROOT/'arms'/os.environ['HRLRVO_ARM']\nSPEC=json.loads((ARM/'config.json').read_text())\nCOPY_SRC=ARM/'source/main/src'\nANCHOR=ROOT/'assets/input/05_anchor_960x540_24fps_1169f.mkv'\nBASELINE=ROOT/'shared/baseline_05_run.readonly_copy.json'")
replace("p.add_argument('--mode',choices=('modified','strength_1p0','control'),required=True)", "p.set_defaults(mode=SPEC['name'], rebase_temporal_position=True, use_latent_carry=SPEC['use_latent_carry'], generated_prefix_strength=1.0)")
replace("p.add_argument('--end-chunk',type=int,choices=(4,19,23),required=True)","p.add_argument('--end-chunk',type=int,choices=(23,),default=23)")
for text in ["    p.add_argument('--rebase-temporal-position',action=argparse.BooleanOptionalAction,default=False)\n", "    p.add_argument('--use-latent-carry',action=argparse.BooleanOptionalAction,default=False)\n", "    p.add_argument('--generated-prefix-strength',type=float,default=1.0)\n"]:replace(text,'')
replace("if ROOT.name!='longrange_fhd_recurrence_test_20260923' or not Path(inf.__file__).resolve().is_relative_to(COPY_SRC):", "if not Path(inf.__file__).resolve().is_relative_to(COPY_SRC.resolve()):")
replace("settings=deepcopy(baseline['settings'])", "settings=json.loads((ARM/'settings.json').read_text())")
replace("checkpoint=Path(baseline['checkpoint_argument'])", "checkpoint=ROOT/'shared/checkpoint-007500'")
replace("outputs=ROOT/'outputs'/a.mode;segments=ROOT/'chunks'/a.mode;latent_root=ROOT/'latent_states'/a.mode", "outputs=ARM/'outputs';segments=ARM/'chunks';latent_root=ARM/'latent_states'")
replace("    store=SegmentedOutputStore(segments)", "    identity_request['intervention']=SPEC\n    identity_request['source_tree_sha256']=json.loads((ARM/'source_manifest.json').read_text())\n    identity_request['settings_sha256']=file_sha256(ARM/'settings.json')\n    identity_request['runner_sha256']=file_sha256(Path(__file__))\n    identity_request['asset_manifest_sha256']=file_sha256(ROOT/'shared/reference_assets.json')\n    store=SegmentedOutputStore(segments)")
replace("atomic_json(ROOT/'configs'/f'effective_{a.mode}_settings.json',settings)","atomic_json(ARM/'effective_settings.json',settings)")
replace("checkpoint_path=checkpoint,memory_optimization=memory)","checkpoint_path=checkpoint,memory_optimization=memory,early_global_boost=SPEC['early_global_boost'])")
replace("    checkpoint=ROOT/'shared/checkpoint-007500'", "    if config.seed!=SPEC['base_seed'] or config.inference_steps!=30:raise RuntimeError('Pinned seed/steps mismatch')\n    if (config.video_cfg_scale,config.video_stg_scale,config.guidance_rescale,list(config.stg_blocks))!=(3.0,1.0,0.7,[28]):raise RuntimeError('Pinned guidance changed')\n    atomic_json(ARM/'effective_sampling.json', {'seed':config.seed,'seed_rule':'base_seed + absolute_start_frame','steps':config.inference_steps,'cfg_scale':config.video_cfg_scale,'stg_scale':config.video_stg_scale,'guidance_rescale':config.guidance_rescale,'stg_blocks':list(config.stg_blocks),'positive_prompt':baseline['positive_prompt'],'negative_prompt':baseline['negative_prompt']})\n    checkpoint=ROOT/'shared/checkpoint-007500'")
(root/'run_arm.py').write_text(s)
old=root.parent/'longrange_fhd_recurrence_test_20260923'
(root/'shared/reference_chunks').mkdir(exist_ok=True)
for p in (old/'chunks/strength_1p0').glob('chunk_*.json'):shutil.copy2(p,root/'shared/reference_chunks'/p.name)
preview=(old/'scripts/watch_chunk_previews.py').read_text()
preview=preview.replace("p.add_argument('--mode',default='modified');", "p.add_argument('--arm',required=True);")
preview=preview.replace("ROOT/'chunks'/a.mode", "ROOT/'arms'/a.arm/'chunks'").replace("ROOT/'outputs'/a.mode/'chunk_previews'", "ROOT/'arms'/a.arm/'outputs/chunk_previews'")
preview=preview.replace("default=4", "default=23")
(root/'tools/watch_chunk_previews.py').write_text(preview)
for gpu,name in enumerate(['01_baseline','02_no_carry','03_global_boost','04_native_halo']):
 spec={'name':name,'physical_gpu':gpu,'base_seed':71634910,'seed_rule':'base_seed + absolute_start_frame','generated_prefix_strength':1.0,'rebase_temporal_position':True,'use_latent_carry':name!='02_no_carry','early_global_boost':name=='03_global_boost','global_boost_scale':1.4 if name=='03_global_boost' else 1.0,'global_boost_steps':10 if name=='03_global_boost' else 0,'global_boost_schedule':'constant_first_10_then_off' if name=='03_global_boost' else 'off','native_halo_radius_tokens':1 if name=='04_native_halo' else 0,'native_token_rgb_stride':32,'anchor_prefix_authority':'current_anchor_encoding','start_policy':'fresh_C0','end_chunk':23}
 (root/'arms'/name/'config.json').write_text(json.dumps(spec,indent=2)+'\n')
