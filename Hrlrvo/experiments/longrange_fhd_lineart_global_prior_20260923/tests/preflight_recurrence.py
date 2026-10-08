"""CPU-only FHD geometry, conditioning, temporal and stitch preflight."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from ltx_outpaint import inference as inf
from ltx_outpaint.geometry import Rect,global_cells,local_windows,anchor_patch,adaptive_layout
from ltx_core.types import SpatioTemporalScaleFactors,VideoLatentShape
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning.types.mask_cond import VideoConditionByMask
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae.video_vae import map_temporal_interval_to_latent
from ltx_core.tools import VideoLatentTools
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_trainer.validation_runner import ValidationRunner

root=Path(__file__).resolve().parents[1]
copy_src=Path(os.environ.get('LINEART_SOURCE_ROOT', str(root/'source/VideoOutpainting_test/main/src')))
module=Path(inf.__file__).resolve()
assert module.is_relative_to(copy_src),(module,copy_src)
assert Path.cwd()==root
sf=SpatioTemporalScaleFactors(time=8,height=32,width=32)
b=inf._vae_boundary(1080,1920,32,32,height_alignment_pixels=128)
assert (b.requested_width,b.requested_height,b.encoded_width,b.encoded_height,b.top,b.left)==(1920,1080,1920,1152,36,0)
rect=inf._center_anchor(b,540,960)
assert rect==(306,480,846,1440)
H,W=b.encoded_height//32,b.encoded_width//32
shape=VideoLatentShape(1,128,9,H,W)
assert (W,H)==(60,36)
a=Rect(9,15,27,45)
assert len(global_cells(H,W,4))==135
counts={}
for k in (4,5):
    windows=local_windows(H,W,k)
    for target in windows:
        anchor_patch(a,target)
        adaptive_layout(a,target,H,W,4)
    counts[f'K{k}']=len(windows)
assert counts=={'K4':16,'K5':25}

def fake_pad(frames,**pad):
    return torch.empty((1,3,frames.shape[0],frames.shape[2]+pad['top']+pad['bottom'],frames.shape[3]+pad['left']+pad['right']),device='meta')
def fake_encode(frames,*args,**kwargs):
    if frames.shape[2]==65:
        assert frames.shape==(1,3,65,576,960)
        return torch.ones((1,128,9,18,30),dtype=torch.bfloat16)
    if frames.shape[2]==1:
        assert frames.shape==(1,3,1,1152,1920)
        return torch.full((1,128,1,H,W),4,dtype=torch.bfloat16)
    assert frames.shape==(1,3,17,1152,1920)
    return torch.full((1,128,3,H,W),2,dtype=torch.bfloat16)
original_pad,original_encode=inf._reflect_pad_rgb,inf._encode_condition_rgb
inf._reflect_pad_rgb=fake_pad;inf._encode_condition_rgb=fake_encode
try:
    anchor=torch.empty((65,3,540,960),device='meta')
    prefix=torch.empty((17,3,1080,1920),device='meta')
    carry=torch.full((1,128,3,H,W),3,dtype=torch.bfloat16)
    condition,actual=inf._condition_latents(anchor,prefix,b,rect,shape,sf,None,None,torch.device('cpu'),torch.bfloat16,prefix_latent_override=carry,generated_prefix_strength=.8)
    assert actual==a
    assert abs(float(condition[0,0,0,12,0])-3.2)<.02
    assert abs(float(condition[0,0,0,12,59])-3.2)<.02
    assert abs(float(condition[0,0,1,12,0])-2.4)<.02
    assert abs(float(condition[0,0,2,12,59])-2.4)<.02
    assert condition[0,0,0,12,30]==1
    assert condition[0,0,0,0,30]==4
    assert condition[0,0,1,0,30]==3
    assert condition[0,0,3,12,0]==0
    assert torch.equal(carry,torch.full_like(carry,3))
finally:
    inf._reflect_pad_rgb=original_pad;inf._encode_condition_rgb=original_encode
mask=inf._known_mask(batch=1,latent_frames=9,latent_height=H,latent_width=W,anchor_rect=a,prefix_frames=17,temporal_factor=8,device=torch.device('cpu'))
assert bool(mask[:,:3].all()) and bool(mask[:,3:,a.y0:a.y1,a.x0:a.x1].all())
assert not bool(mask[:,3:,a.y0:a.y1,:a.x0].any())
tools=VideoLatentTools(patchifier=VideoLatentPatchifier(patch_size=1),target_shape=shape,fps=24,scale_factors=sf,causal_fix=True)
state=tools.create_initial_state(device='cpu',dtype=torch.bfloat16)
state=VideoConditionByMask(condition,mask,strength=1.0).apply_to(state,tools)
known=state.denoise_mask.squeeze(-1)==0
assert int(known.sum())==int(mask.sum())
noised=GaussianNoiser(generator=torch.Generator(device='cpu').manual_seed(123))(state,noise_scale=1.0)
assert torch.equal(noised.latent[known],state.clean_latent[known])
modality=ValidationRunner._modality_from_latent_state(noised,torch.zeros(1,1,1),torch.tensor(.5))
assert bool((modality.timesteps[known]==0).all())
assert bool((modality.timesteps[~known]==.5).all())
assert float(state.positions[:,0].min())==0.0
assert float(state.positions[:,0].max())<3.0
assert tuple(map_temporal_interval_to_latent(48,65,0,0,8)[0].indices(9)[:2])==(6,9)
assert tuple(map_temporal_interval_to_latent(0,17,0,0,8)[0].indices(9)[:2])==(0,3)
assert float(LTX2Scheduler().execute(steps=30)[-1])==0.0
for n,total in ((4,257),(19,977),(23,1169)):
    records=list(inf._iter_long_range_chunks(total,inf.LongRangeSpec(),temporal_factor=8))
    assert len(records)==n+1 and sum(r['output_appended_count'] for r in records)==total
    assert records[0]['output_appended_count']==65
    assert all(r['output_appended_count']==48 for r in records[1:])
    assert records[-1]['new_output_frame_range'][1]==total
one_frame=torch.zeros((1,3,1152,1920),dtype=torch.uint8)
assert one_frame[:,:,b.top:b.bottom,b.left:b.right].shape==(1,3,1080,1920)
print(json.dumps({'status':'PASS','module_file':str(module),'pythonpath':os.environ.get('PYTHONPATH'),'cwd':os.getcwd(),'visible':[1920,1080],'encoded':[1920,1152],'latent':[60,36],'anchor_pixel_yxyx':list(rect),'anchor_latent_yxyx':[a.y0,a.x0,a.y1,a.x1],'global_C4_cells':135,'local_windows':counts,'known_prefix_denoise_mask':0,'known_prefix_timestep':0,'scheduler_final_sigma':0,'stitch_counts':{'C4':257,'C19':977,'C23':1169},'causal_handoff':'first prefix keyframe re-encoded from one RGB frame; remaining two slices direct final-clean-latent carry'},sort_keys=True))
