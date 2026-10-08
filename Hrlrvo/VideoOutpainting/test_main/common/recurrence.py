"""Experiment-only recurrence ablations; production functions remain on disk."""
from dataclasses import replace
import math
import time
import torch


def install(inference, spec):
    originals = {name: getattr(inference, name) for name in
                 ('sample_chunk', '_condition_latents', '_known_mask', '_vae_boundary')}
    context = {'carry': None, 'chunk': 0, 'absolute_start': 0, 'events': []}
    filter_spec = spec.get('prefix_static_filter')
    if filter_spec is not None and not isinstance(filter_spec, dict):
        raise ValueError('prefix_static_filter must be a configuration dictionary')

    def record(action, **values):
        context['events'].append({'chunk': context['chunk'], 'action': action, **values})
    if spec.get('reduced_geometry'):
        def boundary(height, width, hf, wf):
            # Complete C4 latent cells are a v8 model contract, not optional.
            eh = math.ceil(height / (hf * 4)) * hf * 4
            ew = math.ceil(width / (wf * 4)) * wf * 4
            return inference.PixelBoundary(height, width, eh, ew,
                                           (eh-height)//2, (ew-width)//2)
        inference._vae_boundary = boundary

    if spec.get('reassert_anchor_latents'):
        if not hasattr(inference, '_encode_condition_rgb'):
            raise AttributeError(
                'reassert_anchor_latents requires inference._encode_condition_rgb'
            )
        originals['_encode_condition_rgb'] = inference._encode_condition_rgb

        def encode_condition_rgb(*args, **kwargs):
            encoded = originals['_encode_condition_rgb'](*args, **kwargs)
            if kwargs.get('training_aligned_anchor') is True:
                if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
                    raise RuntimeError(
                        'Captured independent Anchor latent must be a 5D tensor'
                    )
                # The native helper already returns this tensor on CPU. Keep the
                # same storage until _condition_latents has copied it to device.
                context['_anchor_latent_capture'] = encoded.detach().cpu()
            return encoded

        inference._encode_condition_rgb = encode_condition_rgb

    def condition(*args, **kwargs):
        context.pop('_anchor_latent_capture', None)
        previous = args[1] if len(args) > 1 else kwargs['previous_prefix']
        if previous is not None and filter_spec is not None:
            from prefix_static_filter import filter_generated_prefix
            boundary = args[2] if len(args) > 2 else kwargs['boundary']
            pixel_rect = args[3] if len(args) > 3 else kwargs['anchor_rect']
            y0, x0, y1, x1 = pixel_rect
            visible_rect = (y0-boundary.top, x0-boundary.left,
                            y1-boundary.top, x1-boundary.left)
            started = time.perf_counter()
            filtered, diagnostics = filter_generated_prefix(
                previous, visible_rect, **filter_spec
            )
            diagnostics['cpu_seconds'] = time.perf_counter()-started
            # Change only the tensor fed to condition encoding. sample_chunk
            # retains the original FP32 history for its exact RGB overwrite.
            if len(args) > 1:
                arguments = list(args)
                arguments[1] = filtered
                args = tuple(arguments)
            else:
                kwargs = {**kwargs, 'previous_prefix': filtered}
            record('prefix_static_filter', **diagnostics)
        values, rect = originals['_condition_latents'](*args, **kwargs)
        if spec.get('reassert_anchor_latents'):
            captured = context.pop('_anchor_latent_capture', None)
            if captured is None:
                raise RuntimeError(
                    'Independent Anchor encoding was not captured during conditioning'
                )
            if previous is not None:
                scale_factors = args[5] if len(args) > 5 else kwargs['scale_factors']
                prefix_latent_count = 1 + (previous.shape[0] - 1) // scale_factors.time
                anchor_height = rect.y1 - rect.y0
                anchor_width = rect.x1 - rect.x0
                expected_capture = (
                    values.shape[0], values.shape[1], values.shape[2],
                    anchor_height, anchor_width,
                )
                if tuple(captured.shape) != expected_capture:
                    raise RuntimeError(
                        'Captured independent Anchor latent does not match conditioning geometry: '
                        f'{tuple(captured.shape)} != {expected_capture}'
                    )
                if not (0 < prefix_latent_count <= values.shape[2]):
                    raise RuntimeError('Prefix latent count is outside the conditioning timeline')
                if not (0 <= rect.y0 < rect.y1 <= values.shape[-2]
                        and 0 <= rect.x0 < rect.x1 <= values.shape[-1]):
                    raise RuntimeError('Anchor rectangle is outside conditioning geometry')
                target = values[
                    :, :, :prefix_latent_count,
                    rect.y0:rect.y1, rect.x0:rect.x1,
                ]
                replacement = captured[:, :, :prefix_latent_count].to(
                    device=values.device, dtype=values.dtype
                )
                delta_rmse = (
                    replacement.float() - target.float()
                ).square().mean().sqrt().item()
                if not math.isfinite(delta_rmse):
                    raise RuntimeError(
                        'Reasserted Anchor latent discrepancy must be finite'
                    )
                target.copy_(replacement)
                record(
                    'reassert_anchor_latents',
                    delta_rmse=delta_rmse,
                    changed=delta_rmse>0,
                    prefix_latent_count=prefix_latent_count,
                    rect={
                        'y0': rect.y0, 'x0': rect.x0,
                        'y1': rect.y1, 'x1': rect.x1,
                    },
                )
        if previous is not None and spec.get('selective_carry') and context['carry'] is not None:
            carry = context['carry'].to(values)
            # Keep the new causal first slice re-encoded. Only the two 8f slices
            # are candidates for carry; preserve independently observed Anchor.
            mask = torch.ones(values.shape[-2:], dtype=torch.bool, device=values.device)
            mask[rect.y0:rect.y1, rect.x0:rect.x1] = False
            current = values[:, :, 1:3, mask]
            delta = carry[:, :, -2:, mask].float() - current.float()
            weight = float(spec.get('selective_carry_weight', 1.0))
            if not 0 <= weight <= 1:
                raise ValueError('selective_carry_weight must be in [0,1]')
            cap = spec.get('carry_relative_cap')
            if cap is not None:
                # Bound the discrepancy per spatial cell using channels/time.
                rms = current.float().square().mean(dim=(1, 2), keepdim=True).sqrt()
                delta_rms = delta.square().mean(dim=(1, 2), keepdim=True).sqrt()
                gate = (float(cap) * rms / delta_rms.clamp_min(1e-6)).clamp(max=1)
                delta = delta * gate
            applied = delta * weight
            values[:, :, 1:3, mask] = (current.float() + applied).to(values.dtype)
            record('selective_carry', weight=weight, relative_cap=cap,
                   applied_rmse=applied.square().mean().sqrt().item(),
                   encoded_rms=current.float().square().mean().sqrt().item())
        return values, rect
    inference._condition_latents = condition

    def known(**kwargs):
        mask = originals['_known_mask'](**kwargs)
        n = kwargs['prefix_frames']
        if n and spec.get('prefix_known_slices') is not None:
            rect = kwargs['anchor_rect']
            end = 1 + (n - 1) // kwargs['temporal_factor']
            begin = int(spec['prefix_known_slices'])
            mask[:, begin:end] = 0
            mask[:, :, rect.y0:rect.y1, rect.x0:rect.x1] = 1
            record('prefix_known_slices', retained_slices=begin, original_slices=end)
        return mask
    inference._known_mask = known

    def chunk(**kwargs):
        context['absolute_start'] = kwargs['absolute_start_frame']
        context['chunk'] = kwargs['absolute_start_frame']//48
        runner = kwargs['runner']
        previous = kwargs['previous_prefix']
        if previous is not None and spec.get('refresh_anchor_rgb'):
            # Correct observed prefix from source while keeping generated RGB.
            previous = previous.clone()
            boundary = kwargs['boundary']
            y0,x0,y1,x1 = kwargs['anchor_rect']
            original_region = previous[:,:,y0-boundary.top:y1-boundary.top,
                                       x0-boundary.left:x1-boundary.left]
            mae = (original_region - kwargs['anchor_frames'][:len(previous)]).abs().mean().item()
            previous[:,:,y0-boundary.top:y1-boundary.top,
                     x0-boundary.left:x1-boundary.left] = kwargs['anchor_frames'][:len(previous)]
            kwargs['previous_prefix'] = previous
            record('refresh_anchor_rgb', source_delta_mae=mae)
        period = spec.get('reset_period_chunks')
        if previous is not None and period and context['chunk'] % period == 0:
            kwargs['previous_prefix'] = None
            context['carry'] = None
            record('periodic_reset', period=period)
        if previous is not None and spec.get('cut_reset'):
            frames=kwargs['anchor_frames']
            delta=(frames[17]-frames[16]).abs().mean().item() if len(frames)>17 else 0
            if delta>float(spec.get('cut_threshold', 0.18)):
                kwargs['previous_prefix']=None
                context['carry']=None
                record('cut_reset', source_delta_mae=delta)
        original_denoise = runner._run_denoising
        original_finalize = runner._finalize_modality
        def denoise(*args, **kw):
            if spec.get('temporal_rebase'):
                shift=context['absolute_start']/runner._config.frame_rate
                positions=kw['video_state'].positions.clone()
                positions[:,0] -= shift
                kw['video_state']=replace(kw['video_state'], positions=positions)
                kw['video_clean']=replace(kw['video_clean'], positions=positions)
                record('temporal_rebase', seconds_removed=shift)
            return original_denoise(*args, **kw)
        def finalize(state, tools, decode_fn, device):
            if spec.get('selective_carry') and state is not None:
                clean=tools.unpatchify(tools.clear_conditioning(state))
                context['carry']=clean.latent[:,:,-2:].detach().cpu().clone()
            return original_finalize(state, tools, decode_fn, device)
        runner._run_denoising=denoise
        runner._finalize_modality=finalize
        try:
            return originals['sample_chunk'](**kwargs)
        finally:
            runner._run_denoising=original_denoise
            runner._finalize_modality=original_finalize
    inference.sample_chunk=chunk
    return context
