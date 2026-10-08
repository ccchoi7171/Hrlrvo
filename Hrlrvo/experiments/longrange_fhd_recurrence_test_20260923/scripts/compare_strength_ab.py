"""CPU-only exact-frame 0.8 vs 1.0 FHD comparison; no pixel quality metric."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
FRAMES=[int(t*24) for t in (10,20,25,28,30,32,34,36,38,40,42,44,46,48)]
LEFT=(0,0,480,1080);RIGHT=(1440,0,1920,1080)
out=ROOT/'comparisons/strength_AB_C0_C23';out.mkdir(parents=True,exist_ok=True)
sheets=ROOT/'contact_sheets/strength_AB_C0_C23';sheets.mkdir(parents=True,exist_ok=True)
for reg,box in [('full',(0,0,1920,1080)),('left',LEFT),('right',RIGHT)]:
    items=[]
    for frame in FRAMES:
        imgs=[]
        for mode in ('modified','strength_1p0'):
            path=ROOT/'frames'/mode/'C0_C23'/'modified'/f'frame_{frame:06d}.jpg'
            if not path.exists():raise FileNotFoundError(path)
            imgs.append(Image.open(path).crop(box))
        pair=Image.new('RGB',(imgs[0].width*2,imgs[0].height+30),'#222')
        pair.paste(imgs[0],(0,30));pair.paste(imgs[1],(imgs[0].width,30))
        ImageDraw.Draw(pair).text((6,7),f'F{frame} {frame/24:.3f}s | 0.8 left / 1.0 right | {reg}',fill='yellow')
        path=out/reg/f'frame_{frame:06d}.jpg';path.parent.mkdir(parents=True,exist_ok=True);pair.save(path,quality=92,subsampling=0)
        width=900 if reg=='full' else 540
        items.append(pair.resize((width,round(pair.height*width/pair.width)),Image.Resampling.LANCZOS))
    cols=2;rows=(len(items)+1)//2;h=max(im.height for im in items)+25;w=items[0].width
    sheet=Image.new('RGB',(cols*w,rows*h),'#222');draw=ImageDraw.Draw(sheet)
    for i,(frame,im) in enumerate(zip(FRAMES,items)):
        x=(i%2)*w;y=(i//2)*h
        draw.text((x+5,y+4),f'{frame/24:.1f}s F{frame}',fill='yellow');sheet.paste(im,(x,y+25))
    sheet.save(sheets/f'timeline_{reg}.jpg',quality=90,subsampling=0)
(ROOT/'report/strength_AB_frames.json').write_text(json.dumps({'global_frames':FRAMES,'0p8':'modified','1p0':'strength_1p0','pixel_quality_metric_used':False},indent=2)+'\n')
print(json.dumps({'status':'complete','frames':FRAMES,'pairs':str(out),'sheets':str(sheets)}))
