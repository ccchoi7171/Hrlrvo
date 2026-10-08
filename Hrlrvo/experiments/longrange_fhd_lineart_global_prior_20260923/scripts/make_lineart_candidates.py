#!/usr/bin/env python3
"""Create native-resolution thin Canny line-art candidates from exact C0 frame."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923')
FRAME_PATH = ROOT / 'reference/c0_frame000.png'
COARSE_PATH = ROOT / 'reference/c0_frame000_lineart.png'
OUT = ROOT / 'reference/candidates/lineart'
DENSITY_NORMALIZER = 0.05

CANDIDATES = {
    'balanced': {
        'clahe': None, 'gaussian_kernel': 5, 'gaussian_sigma': 1.2,
        'canny_low': 50, 'canny_high': 140, 'min_component_pixels': 20, 'min_component_span': 18,
    },
    'detailed': {
        'clahe': None, 'gaussian_kernel': 3, 'gaussian_sigma': 0.8,
        'canny_low': 35, 'canny_high': 100, 'min_component_pixels': 12, 'min_component_span': 12,
    },
    'very_detailed': {
        'clahe': None, 'gaussian_kernel': 3, 'gaussian_sigma': 0.5,
        'canny_low': 20, 'canny_high': 70, 'min_component_pixels': 8, 'min_component_span': 8,
    },
    'clahe_lowlight': {
        'clahe': {'clip_limit': 1.5, 'tile_grid_size': [8, 8]},
        'gaussian_kernel': 5, 'gaussian_sigma': 1.0,
        'canny_low': 55, 'canny_high': 145, 'min_component_pixels': 18, 'min_component_span': 16,
    },
}

CROPS = {
    'leftwindow': [0, 320, 480, 850],
    'rightwindow': [1440, 320, 1920, 850],
    'centerstage': [640, 260, 1280, 900],
    'floor': [480, 760, 1440, 1080],
}


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def rgb_read(path: Path) -> np.ndarray:
    im=cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def rgb_write(path: Path, rgb: np.ndarray) -> None:
    ok=cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
    if not ok: raise RuntimeError(path)


def filter_components(edges: np.ndarray, min_pixels: int, min_span: int):
    n, lab, stats, _=cv2.connectedComponentsWithStats(edges,8)
    keep=np.zeros(n,dtype=bool)
    for k in range(1,n):
        _,_,w,h,a=(int(x) for x in stats[k])
        keep[k] = a >= min_pixels and max(w,h) >= min_span
    return np.where(keep[lab],255,0).astype(np.uint8), {'total':n-1,'kept':int(keep.sum())}


def label_panel(gray: np.ndarray, label: str, width=480, height=270) -> np.ndarray:
    p=cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA)
    if p.ndim==2: p=cv2.cvtColor(p,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(p,(0,0),(width,30),(20,20,20),-1)
    cv2.putText(p,label,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    return p


def labeled_crop(gray: np.ndarray, rect, label: str, size=(360,240)) -> np.ndarray:
    x0,y0,x1,y1=rect
    crop=gray[y0:y1,x0:x1]
    p=cv2.resize(crop,size,interpolation=cv2.INTER_NEAREST)
    if p.ndim==2: p=cv2.cvtColor(p,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(p,(0,0),(size[0],28),(24,24,24),-1)
    cv2.putText(p,label,(7,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    return p


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame=rgb_read(FRAME_PATH)
    if frame.shape != (1080,1920,3): raise ValueError(frame.shape)
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    results={}
    maps={}

    for name,p in CANDIDATES.items():
        work=gray.copy()
        if p['clahe']:
            c=p['clahe']; work=cv2.createCLAHE(c['clip_limit'],tuple(c['tile_grid_size'])).apply(work)
        k=p['gaussian_kernel']
        work=cv2.GaussianBlur(work,(k,k),p['gaussian_sigma'],borderType=cv2.BORDER_REFLECT_101)
        raw=cv2.Canny(work,p['canny_low'],p['canny_high'],apertureSize=3,L2gradient=True)
        edge,comp=filter_components(raw,p['min_component_pixels'],p['min_component_span'])
        # No resize, morphology, dilation, or thinning: retained Canny pixels remain native one-pixel lines.
        binary=(edge>0).astype(np.float32)
        maps[name]=edge
        np.save(OUT/f'{name}.npy',binary,allow_pickle=False)
        cv2.imwrite(str(OUT/f'{name}.png'),edge,[cv2.IMWRITE_PNG_COMPRESSION,3])
        cv2.imwrite(str(OUT/f'{name}_black_on_white.png'),255-edge,[cv2.IMWRITE_PNG_COMPRESSION,3])
        overlay=frame.copy(); m=edge>0
        overlay[m]=np.rint(.3*overlay[m]+.7*np.array([0,255,64])).astype(np.uint8)
        rgb_write(OUT/f'{name}_overlay.png',overlay)

        candidate_crop_dir=OUT/'crops'/name
        candidate_crop_dir.mkdir(parents=True,exist_ok=True)
        for crop_name,(x0,y0,x1,y1) in CROPS.items():
            cv2.imwrite(str(candidate_crop_dir/f'{crop_name}.png'),edge[y0:y1,x0:x1],[cv2.IMWRITE_PNG_COMPRESSION,3])
            cv2.imwrite(str(candidate_crop_dir/f'{crop_name}_black_on_white.png'),255-edge[y0:y1,x0:x1],[cv2.IMWRITE_PNG_COMPRESSION,3])

        padded=np.pad(binary,((36,36),(0,0)),constant_values=0)
        latent=padded.reshape(36,32,60,32).mean(axis=(1,3),dtype=np.float64).astype(np.float32)
        density=latent.reshape(9,4,15,4).mean(axis=(1,3),dtype=np.float64).astype(np.float32)
        strength=np.clip(density/DENSITY_NORMALIZER,0,1).astype(np.float32)
        np.save(OUT/f'{name}_global_line_density.npy',density,allow_pickle=False)
        np.save(OUT/f'{name}_global_line_strength.npy',strength,allow_pickle=False)
        heat=cv2.applyColorMap(np.rint(strength*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)
        heat=cv2.resize(heat,(960,576),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT/f'{name}_global_line_strength_heatmap.png'),heat,[cv2.IMWRITE_PNG_COMPRESSION,3])
        results[name]={
            'parameters':p,
            'component_stats':comp,
            'raw_edge_fraction':float((raw>0).mean()),
            'filtered_native_edge_fraction':float(binary.mean()),
            'global_density':{'min':float(density.min()),'max':float(density.max()),'mean':float(density.mean())},
            'global_strength':{
                'min':float(strength.min()),'max':float(strength.max()),'mean':float(strength.mean()),'std':float(strength.std()),
                'zero_cells':int((strength==0).sum()),'nonzero_cells':int((strength>0).sum()),'one_cells':int((strength==1).sum()),
                'percentiles':{str(q):float(np.percentile(strength,q)) for q in [0,10,25,50,75,90,100]},
            },
        }

    # Full view: earlier coarse reference is included only as a labeled comparison panel.
    coarse=cv2.imread(str(COARSE_PATH),cv2.IMREAD_GRAYSCALE)
    full_panels=[label_panel(coarse,'earlier coarse reference (read-only)')]
    full_panels += [label_panel(maps[n],n) for n in CANDIDATES]
    cv2.imwrite(str(OUT/'contact_sheet_fullview.png'),np.hstack(full_panels),[cv2.IMWRITE_PNG_COMPRESSION,3])

    # Cropped comparison: four structure-relevant crops per candidate; coarse row first.
    all_maps=[('coarse_reference',coarse)]+list(maps.items())
    rows=[]
    for name,edge in all_maps:
        rows.append(np.hstack([labeled_crop(edge,CROPS[c],f'{name} / {c}') for c in CROPS]))
    cv2.imwrite(str(OUT/'contact_sheet_crops.png'),np.vstack(rows),[cv2.IMWRITE_PNG_COMPRESSION,3])

    report={
        'source':{'path':str(FRAME_PATH),'sha256':sha(FRAME_PATH),'shape_hwc':list(frame.shape)},
        'constraints':{
            'cpu_only':True,'native_resolution_canny':True,'resize_before_edge_detection':False,
            'dilation':False,'morphology':False,'learned_model':False,
            'note':'Native 1-pixel edge coverage is not directly comparable with the earlier 4-pixel nearest-upsampled coarse coverage.',
        },
        'mapping':{
            'zero_pad_y':[36,36],'encoded_hw':[1152,1920],'latent_hw':[36,60],'global_hw':[9,15],
            'density_normalizer':DENSITY_NORMALIZER,'strength_formula':'clip(global_density / 0.05, 0, 1)',
            'warning':'Pixel-thin edge detail is substantially collapsed by 128x128-pixel Global cells; g_i*g_j retains only per-cell density, not line connectivity/orientation.',
        },
        'crops_xyxy':CROPS,
        'candidates':results,
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.glob('*')) if p.is_file() and p.name not in {'parameters_and_stats.json','visual_review.md'}},
    }
    (OUT/'parameters_and_stats.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v['global_strength']|{'edge_fraction':v['filtered_native_edge_fraction']} for k,v in results.items()},indent=2))

if __name__=='__main__': main()
