from types import SimpleNamespace
import torch
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
from ltx_outpaint import inference as inf


def test_hybrid_carry_side_attenuation_and_default_off(monkeypatch):
    sf = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    shape = VideoLatentShape(1, 128, 9, 68, 120)
    boundary = inf._vae_boundary(2160, 3840, 32, 32)
    rect = inf._center_anchor(boundary, 1080, 1920)
    anchor = torch.empty(65, 3, 1080, 1920, device='meta')
    prefix = torch.empty(17, 3, 2160, 3840, device='meta')
    monkeypatch.setattr(inf, '_reflect_pad_rgb', lambda frames, **kwargs: torch.empty((1, 3, frames.shape[0], frames.shape[2]+kwargs['top']+kwargs['bottom'], frames.shape[3]+kwargs['left']+kwargs['right']), device='meta'))
    def fake_encode(frames, *_args, **kwargs):
        if frames.shape[2] == 65:
            return torch.ones((1,128,9,34,60), dtype=torch.bfloat16)
        return torch.full((1,128,3,68,120), 2.0, dtype=torch.bfloat16)
    monkeypatch.setattr(inf, '_encode_condition_rgb', fake_encode)
    args = (anchor, prefix, boundary, rect, shape, sf, None, None, torch.device('cpu'), torch.bfloat16)
    original, extent = inf._condition_latents(*args)
    assert (extent.y0, extent.x0, extent.y1, extent.x1) == (17,30,51,90)
    assert original[0,0,0,0,0] == 2 and original[0,0,0,20,40] == 2
    carried = torch.full((1,128,3,68,120), 3.0, dtype=torch.bfloat16)
    changed, extent = inf._condition_latents(*args, prefix_latent_override=carried, generated_prefix_strength=0.8)
    assert abs(float(changed[0,0,0,20,0])-2.4) < .02
    assert abs(float(changed[0,0,0,20,119])-2.4) < .02
    assert changed[0,0,0,20,40] == 1  # independently encoded anchor wins
    assert changed[0,0,0,0,40] == 3  # top, not left/right, remains unchanged
    assert changed[0,0,3,20,0] == 0  # only prefix time is attenuated
    assert torch.equal(carried, torch.full_like(carried, 3))
    mask = inf._known_mask(batch=1, latent_frames=9, latent_height=68, latent_width=120, anchor_rect=extent, prefix_frames=17, temporal_factor=8, device=torch.device('cpu'))
    assert bool(mask[:, :3].all())
    assert bool(mask[:, 3:, 17:51, 30:90].all())
    assert not bool(mask[:, 3:, 17:51, :30].any())


def test_temporal_rebase_changes_only_model_positions(monkeypatch):
    from ltx_outpaint.geometry import Rect
    sf=SpatioTemporalScaleFactors(time=8,height=32,width=32)
    boundary=inf._vae_boundary(64,64,32,32)
    seen=[]
    monkeypatch.setattr(inf,'_condition_latents',lambda *args: (torch.zeros((1,128,9,2,2),dtype=torch.bfloat16),Rect(0,0,1,1)))
    class StopAfterPositions(Exception): pass
    def inspect(**kwargs):
        seen.append(kwargs['video_state'].positions.detach().clone())
        raise StopAfterPositions
    config=SimpleNamespace(frame_rate=24.0,seed=71634910,negative_prompt='',samples=[SimpleNamespace(prompt='')])
    runner=SimpleNamespace(_video_scale_factors=sf,_config=config,_cached_embeddings=[None],_get_prompt_embeddings=lambda *_:(torch.zeros(1,1,1),None,torch.zeros(1,1,1),None),_run_denoising=inspect)
    args=dict(transformer=torch.nn.Linear(1,1,dtype=torch.bfloat16),runner=runner,vae_encoder=None,anchor_frames=torch.empty((65,3,32,32),device='meta'),previous_prefix=None,boundary=boundary,anchor_rect=(16,16,48,48),absolute_start_frame=1008)
    import pytest
    with pytest.raises(StopAfterPositions): inf.sample_chunk(**args,rebase_temporal_position=False)
    with pytest.raises(StopAfterPositions): inf.sample_chunk(**args,rebase_temporal_position=True)
    assert len(seen)==2
    assert torch.allclose(seen[0][:,0]-seen[1][:,0],torch.full_like(seen[0][:,0],42.0))
    assert torch.equal(seen[0][:,1:],seen[1][:,1:])
