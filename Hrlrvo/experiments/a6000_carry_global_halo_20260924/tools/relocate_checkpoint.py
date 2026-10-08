"""Create an isolated path-metadata relocation, preserving every tensor byte."""
from pathlib import Path
import hashlib
import json
import os
import struct

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = ROOT/'assets/models/checkpoint-007500/adapter_model.safetensors'
    target = ROOT/'shared/checkpoint-007500/adapter_model.safetensors'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError('Refusing to overwrite relocated checkpoint: '+str(target))
    with source.open('rb') as src:
        old_length = struct.unpack('<Q', src.read(8))[0]
        before = json.loads(src.read(old_length))
        after = json.loads(json.dumps(before))
        metadata = after['__metadata__']
        metadata['base_model_path'] = str((ROOT/'assets/models/LTX2.3/models/LTX-2.3/ltx-2.3-22b-dev.safetensors').resolve())
        provenance = json.loads(metadata['provenance'])
        provenance['ltx_source_repo'] = str((ROOT/'shared/LTX2.3').resolve())
        metadata['provenance'] = json.dumps(provenance)
        assert {k:v for k,v in before.items() if k!='__metadata__'} == {k:v for k,v in after.items() if k!='__metadata__'}
        header = json.dumps(after, separators=(',', ':'), ensure_ascii=False).encode()
        header += b' '*(-len(header)%8)
        temporary = target.with_suffix('.partial')
        payload_hash = hashlib.sha256()
        with temporary.open('xb') as dst:
            dst.write(struct.pack('<Q', len(header)))
            dst.write(header)
            while data := src.read(8*1024*1024):
                payload_hash.update(data)
                dst.write(data)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, target)
    with target.open('rb') as dst:
        length = struct.unpack('<Q', dst.read(8))[0]
        assert json.loads(dst.read(length)) == after
        verify = hashlib.sha256()
        while data := dst.read(8*1024*1024):
            verify.update(data)
    assert verify.hexdigest() == payload_hash.hexdigest()
    target_hash = hashlib.sha256()
    with target.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            target_hash.update(block)
    receipt = {'target_sha256':target_hash.hexdigest(), 'source':str(source.resolve()), 'target':str(target),
               'tensor_payload_sha256':payload_hash.hexdigest(),
               'tensor_layout_identical':True, 'tensor_bytes_identical':True,
               'metadata_changed_fields':['base_model_path','provenance.ltx_source_repo'],
               'original_metadata':before['__metadata__'], 'runtime_metadata':after['__metadata__']}
    (target.parent/'relocation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Checkpoint tensor payload preserved:', payload_hash.hexdigest())

if __name__ == '__main__':
    main()
