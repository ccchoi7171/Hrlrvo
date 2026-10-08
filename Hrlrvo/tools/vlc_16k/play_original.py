#!/usr/bin/python3
"""Open original 16K media in VLC 3.0.20, retaining its native stream metadata.

VLC's lavc_GetVideoFormat rejects dimensions over 8192 even for CPU decoding.
The installed Ubuntu binaries are pinned by SHA256. Change their four dimension
comparison constants in local copies. Select the local core with LD_LIBRARY_PATH
and the local codec module with a process-local dlopen hook.
No original media or installed VLC binary/configuration is modified.

Usage: play_original.py [VLC options] [media path]
Default: KOCCA Output_2min/01_16K.mp4, 1/16 speed, muted, no skipped frames.
When supplying VLC options, also supply the media path.
Space pauses; Ctrl+J opens codec information; Q exits.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SYSTEM_PLUGIN = Path('/usr/lib/x86_64-linux-gnu/vlc/plugins/codec/libavcodec_plugin.so')
SYSTEM_SHA256 = '0515e302f9462790489ffc8f0f70ac4ba8f4d9b39050fa67dea75321f2ae1e24'
SYSTEM_CORE = Path('/usr/lib/x86_64-linux-gnu/libvlccore.so.9')
CORE_SHA256 = '0d9c5319c56945bad51168cfe144cbec8f3cdb2036394e576f8193c2ebde0908'
INSTALLED_PLUGIN_SHA256 = '4d337fd8194bb7d9716cb8f9532d379a66f9af6041b643881dd4b515db40dafc'
INSTALLED_CORE_SHA256 = '4060d7fae1801c024acf78c9f1765a781bfef36d55b34173f278b85db9b447c9'
DEFAULT_MEDIA = HERE.parents[1] / 'pipeline_8K_to_16K/KOCCA_Verification/Output_2min/01_16K.mp4'


def prepare():
    original = SYSTEM_PLUGIN.read_bytes()
    if hashlib.sha256(original).hexdigest() != SYSTEM_SHA256:
        raise SystemExit('VLC decoder version changed; refusing to patch unknown machine code.')
    patched = bytearray(original)
    # Verified disassembly: cmp $0x2000,%r9d (width), cmp $0x2000,%r10d (height).
    for offset, expected in ((0x4285, bytes.fromhex('4181f900200000')),
                             (0x4299, bytes.fromhex('4181fa00200000'))):
        if original[offset:offset + 7] != expected:
            raise SystemExit('Unexpected dimension-check instruction; refusing to patch.')
        patched[offset + 3:offset + 7] = (16384).to_bytes(4, 'little')
    build = HERE / 'build'
    build.mkdir(exist_ok=True)
    plugin = build / 'libavcodec_plugin.so'
    if not plugin.exists() or plugin.read_bytes() != patched:
        plugin.write_bytes(patched)
    original_core = SYSTEM_CORE.read_bytes()
    if hashlib.sha256(original_core).hexdigest() != CORE_SHA256:
        raise SystemExit('VLC core version changed; refusing to patch unknown machine code.')
    patched_core = bytearray(original_core)
    # VoutValidateFormat: unsigned (dimension - 1) <= 8191, preserving zero checks.
    for offset, expected, immediate in (
        (0x7bf5a, bytes.fromhex('3dff1f0000'), 1),
        (0x7bf74, bytes.fromhex('81faff1f0000'), 2),
    ):
        if original_core[offset:offset + len(expected)] != expected:
            raise SystemExit('Unexpected core dimension-check instruction; refusing to patch.')
        patched_core[offset + immediate:offset + immediate + 4] = (16383).to_bytes(4, 'little')
    core = build / 'libvlccore.so.9'
    if not core.exists() or core.read_bytes() != patched_core:
        core.write_bytes(patched_core)
    hook = build / 'redirect_avcodec.so'
    source = HERE / 'redirect_avcodec.c'
    temporary_hook = build / f'redirect_avcodec.{os.getpid()}.so'
    subprocess.run(['gcc', '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-o', str(temporary_hook), str(source), '-ldl'], check=True)
    temporary_hook.replace(hook)
    return plugin, hook


def desktop_environment():
    env = os.environ.copy()
    if env.get('DISPLAY'):
        return env
    # Native app terminals lack DISPLAY; recover only desktop connection variables.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            if (proc / 'comm').read_text().strip() not in ('vlc', 'kwin_x11'):
                continue
            values = dict(item.split(b'=', 1) for item in (proc / 'environ').read_bytes().split(b'\0')
                          if b'=' in item)
            if b'DISPLAY' not in values:
                continue
            for key in (b'DISPLAY', b'XAUTHORITY', b'DBUS_SESSION_BUS_ADDRESS', b'XDG_RUNTIME_DIR'):
                if key in values:
                    env[key.decode()] = values[key].decode()
            return env
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    raise SystemExit('No desktop DISPLAY found. Run this launcher in the XRDP desktop terminal.')


def main():
    env = desktop_environment()
    installed = (
        hashlib.sha256(SYSTEM_PLUGIN.read_bytes()).hexdigest() == INSTALLED_PLUGIN_SHA256
        and hashlib.sha256(SYSTEM_CORE.read_bytes()).hexdigest() == INSTALLED_CORE_SHA256
    )
    if not installed:
        plugin, hook = prepare()
        env['HRLRVO_VLC_AVCODEC'] = str(plugin)
        env['LD_PRELOAD'] = str(hook) + (' ' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        env['LD_LIBRARY_PATH'] = str(hook.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env['VLC_PLUGIN_PATH'] = str(SYSTEM_PLUGIN.parent.parent)
    args = sys.argv[1:] or [str(DEFAULT_MEDIA)]
    command = ['/usr/bin/vlc', '--no-one-instance', '--no-plugins-cache', '--avcodec-hw=none',
               '--avcodec-threads=4', '--no-avcodec-hurry-up', '--vout=gl', '--gl=glx',
               '--no-drop-late-frames', '--no-skip-frames', '--rate=0.0625',
               '--no-audio', *args]
    os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
