from agents.phi_forecaster import run_forecast

def test_run_forecast_smoke():
    data = [
        {"ds": "2025-05-01", "y": 100},
        {"ds": "2025-05-02", "y": 120},
        {"ds": "2025-05-03", "y": 130},
        {"ds": "2025-05-04", "y": 140},
    ]
    fc = run_forecast(data, future_days=3)
    assert len(fc) == 3
    assert all(k in fc[0] for k in ("ds", "yhat"))
