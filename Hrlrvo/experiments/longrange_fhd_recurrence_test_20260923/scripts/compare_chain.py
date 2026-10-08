"""CPU-only exact-global-frame comparison. Every written file stays in experiment root."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import av
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASELINE=Path('/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_VideoOutPaint_4K_3840x2160/05_VideoOutPaint_4K.mp4')
TIMES=(10,20,25,28,30,32,34,36,38,40,42,44,46,48)
REGIONS={'full':(0,0,1920,1080),'left':(0,0,480,1080),'right':(1440,0,1920,1080)}

def extract(video:Path, frames:set[int], dest:Path):
    dest.mkdir(parents=True,exist_ok=True)
    found=set()
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in frames:continue
            img=Image.fromarray(frame.to_ndarray(format='rgb24'))
            if img.size!=(1920,1080): img=img.resize((1920,1080),Image.Resampling.LANCZOS)
            img.save(dest/f'frame_{index:06d}.jpg',quality=94,subsampling=0)
            found.add(index)
            if found==frames:break
    if found!=frames:raise RuntimeError(f'Missing frames {sorted(frames-found)} from {video}')

def annotated(image:Image.Image, title:str)->Image.Image:
    out=Image.new('RGB',(image.width,image.height+30),'#222222')
    out.paste(image,(0,30))
    ImageDraw.Draw(out).text((6,7),title,fill='white')
    return out

def region_pair(f:int,region:str,frame_root:Path)->Image.Image:
    a=Image.open(frame_root/'baseline'/f'frame_{f:06d}.jpg')
    b=Image.open(frame_root/'modified'/f'frame_{f:06d}.jpg')
    box=REGIONS[region]
    a=a.crop(box);b=b.crop(box)
    pair=Image.new('RGB',(a.width+b.width,a.height),'black')
    pair.paste(a,(0,0));pair.paste(b,(a.width,0))
    return annotated(pair,f'F{f} | {f/24:.3f}s | baseline 4K downsized  /  new FHD | {region}')

def contact(files:list[Path],labels:list[str],dest:Path,cols:int,target_w:int):
    thumbs=[]
    for p in files:
        im=Image.open(p)
        ratio=target_w/im.width
        im=im.resize((target_w,round(im.height*ratio)),Image.Resampling.LANCZOS)
        thumbs.append(im)
    rows=(len(thumbs)+cols-1)//cols
    cell_h=max(im.height for im in thumbs)+24
    sheet=Image.new('RGB',(cols*target_w,rows*cell_h),'#1e1e1e')
    pen=ImageDraw.Draw(sheet)
    for i,(im,label) in enumerate(zip(thumbs,labels)):
        x=(i%cols)*target_w;y=(i//cols)*cell_h
        pen.text((x+4,y+4),label,fill='yellow')
        sheet.paste(im,(x,y+24))
    dest.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(dest,quality=91,subsampling=0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('modified','strength_1p0'),default='modified')
    p.add_argument('--end-chunk',type=int,choices=(4,10,19,23),required=True)
    a=p.parse_args()
    last=65+48*a.end_chunk-1
    mp4=ROOT/'outputs'/a.mode/f'C0_C{a.end_chunk}_{last+1}f_1920x1080.mp4'
    if not mp4.is_file():raise FileNotFoundError(mp4)
    timeline={int(t*24) for t in TIMES if t*24<=last}
    if a.end_chunk==4:timeline.update((0,48,96,144,192,240))
    phases={}
    for c in range(a.end_chunk+1):
        start=c*48
        phases[c]={'start':start,'prefix_last':start+16,'new_first':start+17,'new_middle':start+41,'new_last':start+64}
    selected=timeline|{f for phase in phases.values() for f in phase.values()}
    assert max(selected)<=last
    frames=ROOT/'frames'/a.mode/f'C0_C{a.end_chunk}'
    extract(BASELINE,selected,frames/'baseline')
    extract(mp4,selected,frames/'modified')
    compares=ROOT/'comparisons'/a.mode/f'C0_C{a.end_chunk}'
    for region in REGIONS:
        (compares/region).mkdir(parents=True,exist_ok=True)
        for f in sorted(selected):
            region_pair(f,region,frames).save(compares/region/f'frame_{f:06d}.jpg',quality=92,subsampling=0)
    sheets=ROOT/'contact_sheets'/a.mode/f'C0_C{a.end_chunk}'
    for region in REGIONS:
        files=[compares/region/f'frame_{f:06d}.jpg' for f in sorted(timeline)]
        contact(files,[f'{f/24:.1f}s (F{f})' for f in sorted(timeline)],sheets/f'timeline_{region}.jpg',3,960 if region=='full' else 480)
        for c,points in phases.items():
            order=['start','prefix_last','new_first','new_middle','new_last']
            fs=[points[k] for k in order]
            files=[compares/region/f'frame_{f:06d}.jpg' for f in fs]
            contact(files,[f'C{c} {k} F{f}' for k,f in zip(order,fs)],sheets/f'C{c:02d}_{region}_phases.jpg',5,640 if region=='full' else 300)
    meta={'baseline':str(BASELINE),'modified':str(mp4),'frames_compared':sorted(selected),'timeline_frames':sorted(timeline),'phases':phases,'fps':24,'baseline_resize':[1920,1080],'pixel_metric_used':False}
    (ROOT/'report'/f'comparison_{a.mode}_C0_C{a.end_chunk}.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'status':'complete','end_chunk':a.end_chunk,'frames_compared':len(selected),'comparison_root':str(compares),'contact_root':str(sheets)}))
if __name__=='__main__':main()
