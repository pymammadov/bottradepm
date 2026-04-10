from src.strategy import compute_position_size


def test_risk_sizing_correctness():
    qty = compute_position_size(
        equity=10_000,
        entry_price=50_000,
        stop_distance=1_000,
        risk_pct=0.0075,
        max_leverage=2.0,
    )
    assert round(qty, 8) == 0.075


def test_leverage_cap_does_not_increase_risk():
    qty = compute_position_size(
        equity=10_000,
        entry_price=100_000,
        stop_distance=100,
        risk_pct=0.0075,
        max_leverage=0.5,
    )
    risk = qty * 100
    assert qty == 0.05
    assert risk <= 10_000 * 0.0075
