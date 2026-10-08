"""CPU-only exact-global-frame visual comparison. Writes below EXP only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path('/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_VideoOutPaint_4K_3840x2160/05_VideoOutPaint_4K.mp4')
WINDOWS = {'A': (14,16), 'B': (21,23)}
PHASES = ('start','prefix_last','new_first','new_middle','last')

def load_selected(path, index_map):
    selected = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in index_map:
                selected[index_map[index]] = Image.fromarray(frame.to_ndarray(format='rgb24'))
                if len(selected)==len(index_map): break
    if len(selected)!=len(index_map):
        raise RuntimeError(f'Could not decode every requested frame from {path}')
    return selected

def label(image, title):
    out=Image.new('RGB',(image.width,image.height+28),'#181818')
    out.paste(image,(0,28))
    ImageDraw.Draw(out).text((7,7),title,fill='white',font=ImageFont.load_default())
    return out

def pair(base, modified, region):
    if region=='full':
        base=base.resize((960,540),Image.Resampling.LANCZOS)
        modified=modified.resize((960,540),Image.Resampling.LANCZOS)
    else:
        x0,x1=(0,960) if region=='left' else (2880,3840)
        base=base.crop((x0,0,x1,2160)).resize((480,1080),Image.Resampling.LANCZOS)
        modified=modified.crop((x0,0,x1,2160)).resize((480,1080),Image.Resampling.LANCZOS)
    out=Image.new('RGB',(base.width+modified.width,base.height),'black')
    out.paste(base,(0,0));out.paste(modified,(base.width,0))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--window',choices=WINDOWS,required=True)
    a=p.parse_args()
    start,end=WINDOWS[a.window]
    output=ROOT/'modified_outputs'/f'window_{a.window}'
    mapping=json.loads((output/'video_frame_map.json').read_text())
    first=mapping['global_first_frame'];last=mapping['global_last_frame_inclusive']
    mp4=Path(mapping['mp4'])
    phase_frames={}
    for c in range(start,end+1):
        global_start=c*48
        phase_frames[c]={'start':global_start,'prefix_last':global_start+16,'new_first':global_start+17,'new_middle':global_start+40,'last':global_start+64}
    selected=sorted(set(f for phases in phase_frames.values() for f in phases.values()))
    if selected[0]<first or selected[-1]>last: raise RuntimeError('Phase frame outside modified MP4')
    baseline=load_selected(BASELINE,{f:f for f in selected})
    modified=load_selected(mp4,{f-first:f for f in selected})
    extracts=ROOT/'frame_extracts'/f'window_{a.window}'
    compares=ROOT/'comparisons'/f'window_{a.window}'
    sheets=ROOT/'contact_sheets'/f'window_{a.window}'
    for group in ('baseline','modified'): (extracts/group).mkdir(parents=True,exist_ok=True)
    for region in ('full','left','right'): (compares/region).mkdir(parents=True,exist_ok=True)
    sheets.mkdir(parents=True,exist_ok=True)
    for f in selected:
        baseline[f].save(extracts/'baseline'/f'frame_{f:06d}.png')
        modified[f].save(extracts/'modified'/f'frame_{f:06d}.png')
        for region in ('full','left','right'):
            label(pair(baseline[f],modified[f],region),f'{region.upper()}   global frame {f} / {f/24:.3f}s   BASELINE | MODIFIED').save(compares/region/f'frame_{f:06d}.png')
    for region in ('full','left','right'):
        sample=Image.open(compares/region/f'frame_{selected[0]:06d}.png')
        tw,th=sample.size
        scale=.4 if region=='full' else .35
        iw,ih=round(tw*scale),round(th*scale)
        sheet=Image.new('RGB',(5*iw,3*ih),'#202020')
        for row,c in enumerate(range(start,end+1)):
            for col,phase in enumerate(PHASES):
                f=phase_frames[c][phase]
                im=Image.open(compares/region/f'frame_{f:06d}.png').resize((iw,ih),Image.Resampling.LANCZOS)
                sheet.paste(im,(col*iw,row*ih))
                ImageDraw.Draw(sheet).text((col*iw+5,row*ih+5),f'C{c} {phase} F{f}',fill='yellow',stroke_width=1,stroke_fill='black')
        sheet.save(sheets/f'{region}_phases.png')
    (ROOT/'report'/f'comparison_frames_window_{a.window}.json').write_text(json.dumps({'global_frames':selected,'phase_frames':phase_frames,'baseline':str(BASELINE),'modified':str(mp4),'exact_global_offset':first},indent=2)+'\n')
    print(json.dumps({'window':a.window,'frames':selected,'comparison_root':str(compares),'contact_root':str(sheets)}))

if __name__=='__main__': main()
