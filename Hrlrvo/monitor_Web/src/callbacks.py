"""Dash callbacks backed only by read-only filesystem readers."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from dash import Input, Output, State, html, no_update

from .components import gradient_table, health_panel, ratio_table
from .log_reader import ReaderRegistry, discover_runs, read_last_lines, read_text_artifact
from .metrics import (
    ALIASES, GRADIENT_GROUPS, block_figure, block_records, canonicalize_record,
    display_value, empty_figure, gradient_bar, health_summary, latest_record,
    line_figure, loss_figure, scalar_frame, status_from,
    sustained_zero_gradient_groups,
)


def _bounded_block_options(records, limit=500):
    available=block_records(records)
    if len(available)>limit:
        indices={round(i*(len(available)-1)/(limit-1)) for i in range(limit)}
        available=[available[i] for i in sorted(indices)]
    return [{"label":f"Step {step}","value":str(step)} for step,_ in available]


def _find_block_record(records, selected):
    available=block_records(records)
    if not available: return None
    if selected is not None:
        for step,record in available:
            if str(step)==str(selected): return record
    return available[-1][1]


def register_callbacks(app, *, registry: ReaderRegistry, runs_root: str | None, direct_log: str | None, mock: bool):
    @app.callback(Output("refresh-tick","interval"),Output("refresh-tick","disabled"),Input("refresh-control","value"))
    def set_refresh(value):
        if not value: return no_update,True
        return int(value),False

    @app.callback(Output("run-selector","options"),Output("run-selector","value"),Input("discovery-tick","n_intervals"),State("run-selector","value"))
    def update_runs(_tick,current):
        runs=discover_runs(runs_root,direct_log=direct_log,mock=mock)
        options=[{"label":r["label"],"value":r["value"]} for r in runs]
        values={r["value"] for r in runs}
        if current in values: return options,current
        return options,(runs[0]["value"] if runs else None)

    outputs=[
        Output("run-value","children"),Output("step-value","children"),Output("samples-value","children"),Output("batches-value","children"),
        Output("loss-value","children"),Output("lr-value","children"),Output("status-value","children"),Output("status-card","className"),
        Output("alert-banner","children"),Output("alert-banner","className"),Output("reader-state","children"),
        Output("loss-graph","figure"),Output("lr-graph","figure"),Output("lora-graph","figure"),Output("compression-graph","figure"),
        Output("fusion-grad-graph","figure"),Output("local-ratio-graph","figure"),Output("global-ratio-graph","figure"),
        Output("local-ratio-table","children"),Output("global-ratio-table","children"),Output("health-content","children"),
        Output("gradient-table","children"),Output("gradient-bar","figure"),Output("console-view","children"),
        Output("run-config","children"),Output("run-manifest","children"),Output("block-step","options"),Output("block-step","value"),
    ]

    @app.callback(*outputs,Input("refresh-tick","n_intervals"),Input("run-selector","value"),Input("grad-scale","value"),Input("console-lines","value"),State("block-step","value"))
    def refresh_dashboard(_tick,log_path,grad_scale,console_lines,current_block_step):
        blank=empty_figure("Unavailable","Select a run")
        if not log_path:
            return ("N/A","N/A","N/A","N/A","N/A","N/A","ERROR","status-card status-critical",
                    "No train_metrics.jsonl discovered","alert-banner critical","No active reader",
                    blank,blank,blank,blank,blank,blank,blank,html.Div("N/A"),html.Div("N/A"),html.Div("No data"),
                    gradient_table({}),blank,"console log unavailable","unavailable","unavailable",[],None)
        reader=registry.get(log_path); snapshot=reader.refresh(); records=snapshot.records
        latest=latest_record(records); frame=scalar_frame(records)
        run_name="MOCK DATA" if mock else Path(log_path).parent.name
        status,status_class,age=status_from(latest,snapshot.file_mtime)
        if snapshot.last_error: status,status_class="ERROR","status-critical"
        missing=[key for key in ALIASES if key not in {"timestamp","run_id","local_native_ratio_by_block","local_adaptive_ratio_by_block","global_local_ratio_by_block","global_global_ratio_by_block"} and latest.get(key) is None]
        health=health_summary(latest,missing)
        sustained_zero=sustained_zero_gradient_groups(frame)
        if sustained_zero:
            health["warnings"].append("Gradient norm stayed zero for 10 logged steps: "+", ".join(sustained_zero))
            if health["level"]=="healthy": health["level"]="warning"
        alerts=list(health["critical"])+list(health["warnings"])
        if status in {"STALE","STOPPED / NO UPDATE","ERROR"}: alerts.insert(0,f"Logging status: {status}")
        alert_class="alert-banner critical" if health["critical"] or status=="ERROR" else "alert-banner warning" if alerts else "alert-banner hidden"
        reader_text=(f"Read-only incremental tail: {len(records):,} records | byte offset {snapshot.byte_offset:,} | "
                     f"pending bytes {snapshot.pending_bytes} | malformed lines skipped {snapshot.malformed_lines} | resets {snapshot.reset_count}"+
                     (f" | last update age {age:.1f}s" if age is not None else ""))
        log_y=grad_scale=="log"
        lora=line_figure(frame,{"OP-LoRA":"grad_norm_op_lora","A-LoRA":"grad_norm_a_lora","G-LoRA":"grad_norm_g_lora","F-LoRA":"grad_norm_f_lora"},"LoRA gradient norms",log_y=log_y)
        compression=line_figure(frame,{"C2":"grad_norm_c2","Shared C4":"grad_norm_c4_shared","R2":"grad_norm_r2","R4":"grad_norm_r4"},"Compression / restoration gradient norms",log_y=log_y)
        fusion=line_figure(frame,{"Local Fusion":"grad_norm_local_fusion","Global Fusion":"grad_norm_global_fusion"},"Fusion gradient norms",log_y=log_y)
        local_ratio=line_figure(frame,{"Native":"local_native_ratio_mean","Adaptive":"local_adaptive_ratio_mean"},"Local fusion mean ratio",y_range=[0,1])
        global_ratio=line_figure(frame,{"Local":"global_local_ratio_mean","Global":"global_global_ratio_mean"},"Global fusion mean ratio",y_range=[0,1])
        run_dir=Path(log_path).parent
        options=_bounded_block_options(records)
        values={o["value"] for o in options}
        selected=current_block_step if current_block_step in values else (options[-1]["value"] if options else None)
        return (
            run_name,display_value(latest.get("optimizer_step")),display_value(latest.get("samples_seen")),display_value(latest.get("batches_seen")),
            display_value(latest.get("loss")),display_value(latest.get("learning_rate")),status,f"status-card {status_class}",
            html.Ul([html.Li(x) for x in alerts]),alert_class,reader_text,
            loss_figure(frame),line_figure(frame,{"Learning rate":"learning_rate"},"Learning rate"),lora,compression,fusion,local_ratio,global_ratio,
            ratio_table(latest,[("Native","local_native_ratio"),("Adaptive","local_adaptive_ratio")]),
            ratio_table(latest,[("Local","global_local_ratio"),("Global","global_global_ratio")]),
            health_panel(health,latest),gradient_table(latest),gradient_bar(latest),
            read_last_lines(run_dir/"train_console.log",int(console_lines or 100)),
            read_text_artifact(run_dir/"run_config.yaml"),read_text_artifact(run_dir/"run_manifest.json"),options,selected,
        )

    @app.callback(Output("local-block-graph","figure"),Output("global-block-graph","figure"),Input("block-step","value"),Input("run-selector","value"),Input("refresh-tick","n_intervals"))
    def refresh_blocks(selected,log_path,_tick):
        if not log_path: return empty_figure("Local fusion by block"),empty_figure("Global fusion by block")
        records=registry.get(log_path).snapshot().records
        record=_find_block_record(records,selected)
        return (
            block_figure(record,("Native","local_native_ratio_by_block"),("Adaptive","local_adaptive_ratio_by_block"),"Local fusion: 48 DiT blocks"),
            block_figure(record,("Local","global_local_ratio_by_block"),("Global","global_global_ratio_by_block"),"Global fusion: 48 DiT blocks"),
        )
