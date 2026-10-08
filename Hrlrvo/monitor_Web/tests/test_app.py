from __future__ import annotations
from app import create_app,parse_args


def test_mock_app_layout_health_and_snapshot():
    app=create_app(mock=True)
    client=app.server.test_client()
    assert client.get('/').status_code==200
    assert client.get('/_dash-layout').status_code==200
    deps=client.get('/_dash-dependencies'); assert deps.status_code==200 and len(deps.get_json())>=4
    health=client.get('/healthz').get_json()
    assert health['status']=='ok' and health['mode']=='mock' and health['gpu'] is False
    assert health['monitor_rss_mib']>0
    snap=client.get('/api/snapshot').get_json(); assert snap['status']=='ok' and snap['records']==160
    assert snap['latest']['optimizer_step']==160


def test_mock_dashboard_callbacks_render_all_panels():
    app=create_app(mock=True)
    callbacks=[entry['callback'].__wrapped__ for entry in app.callback_map.values()]
    options,path=callbacks[1](0,None)
    assert len(options)==1 and path.endswith('mock_data/train_metrics.jsonl')
    dashboard=callbacks[2](0,path,'linear',100,None)
    assert len(dashboard)==28
    assert dashboard[1]=='160'
    assert len(dashboard[-2])==160 and dashboard[-1]=='160'
    local,global_=callbacks[3](dashboard[-1],path,0)
    assert len(local.data)==2 and len(global_.data)==2


def test_cli_rejects_relative_direct_log():
    try: parse_args(['--log','relative/train_metrics.jsonl'])
    except SystemExit as exc: assert exc.code==2
    else: raise AssertionError('relative --log must fail')
