#!/usr/bin/env python3
"""Standalone, read-only Hrlrvo training metrics dashboard."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import psutil
from dash import Dash
from flask import jsonify

from src.callbacks import register_callbacks
from src.layout import make_layout
from src.log_reader import ReaderRegistry, discover_runs
from src.metrics import latest_record

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = Path('/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs')
MOCK_LOG = PROJECT_ROOT / 'mock_data/train_metrics.jsonl'


def create_app(*, runs_root: str | None = str(DEFAULT_RUNS_ROOT), direct_log: str | None = None, mock: bool = False, max_records: int = 100_000) -> Dash:
    if mock:
        direct_log = str(MOCK_LOG)
        runs_root = None
    registry = ReaderRegistry(max_records=max_records)
    app = Dash(__name__, title='Hrlrvo Training Monitor', update_title='Refreshing…')
    app.layout = make_layout(mock=mock)
    register_callbacks(app, registry=registry, runs_root=runs_root, direct_log=direct_log, mock=mock)

    @app.server.get('/healthz')
    def healthz():
        process=psutil.Process(os.getpid())
        return jsonify({'status':'ok','mode':'mock' if mock else 'read-only','gpu':False,
                        'monitor_rss_mib':round(process.memory_info().rss/(1024**2),2)})

    @app.server.get('/api/snapshot')
    def api_snapshot():
        runs=discover_runs(runs_root,direct_log=direct_log,mock=mock)
        if not runs:
            return jsonify({'status':'no-log','records':0,'latest':{}})
        snap=registry.get(runs[0]['value']).refresh()
        return jsonify({'status':'ok' if not snap.last_error else 'error','records':len(snap.records),'byte_offset':snap.byte_offset,
                        'pending_bytes':snap.pending_bytes,'malformed_lines':snap.malformed_lines,
                        'latest':latest_record(snap.records),'error':snap.last_error})
    return app


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--log',type=Path,help='Absolute path to one train_metrics.jsonl; overrides discovery')
    source.add_argument('--mock',action='store_true',help='Use bundled synthetic data and never read VideoOutpainting')
    parser.add_argument('--runs-root',type=Path,default=DEFAULT_RUNS_ROOT,help='Read-only auto-discovery root')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8050)
    parser.add_argument('--max-records',type=int,default=100_000,help='Maximum records retained per run in RAM')
    parser.add_argument('--debug',action='store_true')
    args=parser.parse_args(argv)
    if args.log is not None and not args.log.is_absolute(): parser.error('--log must be an absolute path')
    if args.log is not None and args.log.resolve().name!='train_metrics.jsonl':
        parser.error('--log must point to a file named train_metrics.jsonl')
    if not 1 <= args.port <= 65535: parser.error('--port must be in 1..65535')
    if args.max_records < 1: parser.error('--max-records must be positive')
    if args.mock and not MOCK_LOG.is_file(): parser.error(f'mock log unavailable; run scripts/generate_mock.py: {MOCK_LOG}')
    return args


def main():
    args=parse_args()
    app=create_app(runs_root=None if args.mock else str(args.runs_root.expanduser().resolve()),
                   direct_log=str(args.log.expanduser().resolve()) if args.log else None,
                   mock=args.mock,max_records=args.max_records)
    app.run(host=args.host,port=args.port,debug=args.debug)


if __name__=='__main__':
    main()
