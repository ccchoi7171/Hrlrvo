"""CPU-only preview encoder; observes committed segments without touching GPU producer."""
import argparse,json,os,time
from pathlib import Path
import av,numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--end-chunk',type=int,default=23);p.add_argument('--producer-pid',type=int,required=True);a=p.parse_args()
segments=ROOT/'arms'/a.arm/'chunks'
out=ROOT/'arms'/a.arm/'outputs/chunk_previews';out.mkdir(parents=True,exist_ok=True)
for c in range(a.end_chunk+1):
    rec=segments/f'chunk_{c:06d}.json'
    while not rec.exists():
        if not Path(f'/proc/{a.producer_pid}').exists():
            raise RuntimeError(f'Producer exited before C{c} commit')
        time.sleep(10)
    data=json.loads(rec.read_text())
    count=int(data['output_appended_count'])
    if count!=(65 if c==0 else 48):raise RuntimeError('Committed frame count mismatch')
    src=segments/data['segment_file']
    dst=out/f'C{c:02d}_{"full65" if c==0 else "new48"}.mp4'
    if dst.exists():continue
    partial=dst.with_name('.'+dst.stem+'.partial.mp4')
    frames=np.load(src,mmap_mode='r',allow_pickle=False)
    if frames.shape!=(count,1080,1920,3) or frames.dtype!=np.uint8:raise RuntimeError('Segment shape/dtype mismatch')
    with av.open(str(partial),'w') as container:
        stream=container.add_stream('libx264',rate=24)
        stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
        stream.options={'crf':'20','preset':'veryfast','threads':'2'}
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    os.replace(partial,dst)
    if a.arm == '01_baseline':
        reference_path = ROOT/'shared/reference_chunks'/f'chunk_{c:06d}.json'
        reference = json.loads(reference_path.read_text()) if reference_path.is_file() else None
        comparison = {'chunk':c, 'same_chunk_seed':None if reference is None else data['chunk_seed']==reference['chunk_seed'], 'lossless_pixels_sha256_equal':None if reference is None else data['segment_pixel_sha256']==reference['segment_pixel_sha256'], 'reference_available':reference is not None, 'note':'Fresh-C0 A6000 run versus historical bootstrapped-C0 run; mismatch is not proof of code/config drift.'}
        with (ROOT/'arms'/a.arm/'outputs/reproducibility_hashes.jsonl').open('a') as report:
            report.write(json.dumps(comparison)+'\n')
    print(json.dumps({'chunk':c,'preview_mp4':str(dst),'frames':count,'global_frame_range':data['new_output_frame_range']}),flush=True)
print(json.dumps({'status':'complete','chunks':a.end_chunk+1}),flush=True)
