"""Dark engineering dashboard layout."""
from __future__ import annotations

from dash import dcc, html
from .components import section, status_card


def make_layout(*, mock: bool):
    return html.Div(
        [
            dcc.Interval(id="refresh-tick", interval=2000, n_intervals=0),
            dcc.Interval(id="discovery-tick", interval=10000, n_intervals=0),
            html.Header(
                [
                    html.Div([html.H1("Hrlrvo Training Monitor"),html.Div("READ-ONLY FILESYSTEM MONITOR",className="readonly-badge")]),
                    html.Div("MOCK DATA — SYNTHETIC METRICS" if mock else "",className="mock-banner" if mock else "hidden"),
                    html.Div(
                        [
                            html.Div([html.Label("Run"),dcc.Dropdown(id="run-selector",placeholder="No train_metrics.jsonl discovered",clearable=False)],className="control run-control"),
                            html.Div([html.Label("Refresh"),dcc.Dropdown(id="refresh-control",options=[{"label":"1 sec","value":1000},{"label":"2 sec","value":2000},{"label":"5 sec","value":5000},{"label":"10 sec","value":10000},{"label":"Pause","value":0}],value=2000,clearable=False)],className="control"),
                            html.Div([html.Label("Gradient scale"),dcc.RadioItems(id="grad-scale",options=[{"label":"Linear","value":"linear"},{"label":"Log","value":"log"}],value="linear",inline=True)],className="control"),
                        ],className="controls",
                    ),
                ],className="topbar",
            ),
            html.Main(
                [
                    html.Div([
                        status_card("RUN","run-value"),status_card("STEP","step-value"),status_card("SAMPLES","samples-value"),
                        status_card("BATCHES","batches-value"),status_card("LOSS","loss-value"),status_card("LR","lr-value"),
                        status_card("STATUS","status-value",card_id="status-card"),
                    ],className="status-grid"),
                    html.Div(id="alert-banner",className="alert-banner hidden"),
                    html.Div(id="reader-state",className="reader-state"),
                    html.Div([
                        section("Loss",dcc.Graph(id="loss-graph",config={"displaylogo":False})),
                        section("Learning Rate",dcc.Graph(id="lr-graph",config={"displaylogo":False})),
                    ],className="grid two"),
                    html.Div([
                        section("LoRA Gradient",dcc.Graph(id="lora-graph",config={"displaylogo":False})),
                        section("Compression / Restoration Gradient",dcc.Graph(id="compression-graph",config={"displaylogo":False})),
                        section("Fusion Gradient",dcc.Graph(id="fusion-grad-graph",config={"displaylogo":False})),
                    ],className="grid three"),
                    html.Div([
                        section("Local Fusion Actual Ratio",[dcc.Graph(id="local-ratio-graph",config={"displaylogo":False}),html.Div(id="local-ratio-table")]),
                        section("Global Fusion Actual Ratio",[dcc.Graph(id="global-ratio-graph",config={"displaylogo":False}),html.Div(id="global-ratio-table")]),
                    ],className="grid two"),
                    section("48-block Fusion Distribution",[
                        html.Div([html.Label("Optimizer step"),dcc.Dropdown(id="block-step",clearable=False)],className="block-control"),
                        html.Div([dcc.Graph(id="local-block-graph",config={"displaylogo":False}),dcc.Graph(id="global-block-graph",config={"displaylogo":False})],className="grid two inner"),
                    ]),
                    html.Div([
                        section("Training Health",html.Div(id="health-content")),
                        section("Current Gradient Table",html.Div(id="gradient-table")),
                        section("Gradient Magnitude",dcc.Graph(id="gradient-bar",config={"displaylogo":False})),
                    ],className="grid three health-grid"),
                    section("Console Log Viewer",[
                        html.Div([html.Label("Tail lines"),dcc.Dropdown(id="console-lines",options=[50,100,200,500],value=100,clearable=False)],className="console-control"),
                        html.Pre(id="console-view",className="console-view"),
                    ]),
                    html.Details([html.Summary("Run Config (read-only)"),html.Pre(id="run-config",className="artifact-view")],className="details-panel"),
                    html.Details([html.Summary("Run Manifest (read-only)"),html.Pre(id="run-manifest",className="artifact-view")],className="details-panel"),
                ],className="content",
            ),
            html.Footer("Monitor reads appended files only. It cannot start, stop, signal, or modify training."),
        ],className="app-shell",
    )
