"""CPU-only content verification against the original server's exact assets."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(16*1024*1024):
            h.update(block)
    return h.hexdigest()

def asset_path(relative):
    prefix = 'models/LTX2.3/packages/'
    if relative.startswith(prefix):
        return ROOT/'shared/LTX2.3/packages'/relative.removeprefix(prefix)
    return ROOT/'assets'/relative

def main():
    manifest = ROOT/'shared/reference_assets.json'
    expected = json.loads(manifest.read_text())
    # Validate the complete required set before any expensive checksum.
    for relative, info in expected.items():
        path=asset_path(relative)
        if not path.is_file() or path.stat().st_size!=info['bytes']:
            raise RuntimeError('Missing/wrong-size required asset: '+relative)
    verified = {}
    for relative, info in expected.items():
        path = asset_path(relative)
        stat = path.stat()
        if stat.st_size != info['bytes'] or digest(path) != info['sha256']:
            raise RuntimeError('Reference asset mismatch: '+relative)
        after = path.stat()
        if (stat.st_size,stat.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise RuntimeError('Asset changed during hashing: '+relative)
        verified[relative] = {'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns,
                              'sha256':info['sha256']}
        if stat.st_size > 100_000_000:
            print('VERIFIED', relative, info['sha256'], flush=True)
    report = {'passed':True,'manifest_sha256':digest(manifest),'files':verified}
    target = ROOT/'verification/asset_verification.json'
    temporary = target.with_suffix('.partial')
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    os.replace(temporary,target)
    print('ALL_ASSETS_VERIFIED',len(verified),flush=True)

if __name__ == '__main__':
    main()
