from datetime import datetime

import pandas as pd

from recon_engine import _pair_reconcile


def row(key, when, amount, currency="USD", secondary="", merchant=""):
    return {
        "provider": "Test",
        "key": key,
        "secondary_id": secondary,
        "merchant_id": merchant,
        "time_gmt3": pd.Timestamp(when),
        "amount": amount,
        "currency": currency,
        "source_amount": amount,
        "source_currency": currency,
        "channel": "",
        "email": "",
        "entity": "CFD",
        "record_id": "",
        "details": "",
        "source_file": "test.csv",
    }


def test_exact_match_and_amount_mismatch():
    left = pd.DataFrame([
        row("A", "2026-08-14 10:00:00", 100.00),
        row("B", "2026-08-14 11:00:00", 50.00),
    ])
    right = pd.DataFrame([
        row("A", "2026-08-14 10:00:05", 100.00),
        row("B", "2026-08-14 11:00:05", 49.00),
    ])
    result = _pair_reconcile(
        left, right, pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-15"),
        "Test PSP", "PSP", "Orchestrator", 0.02,
    )
    assert result["summary"]["Matched"] == 1
    assert result["summary"]["Amount Mismatches"] == 1
    assert result["summary"]["Action Exceptions"] == 1


def test_counterpart_outside_period_is_not_missing():
    left = pd.DataFrame([row("A", "2026-08-14 10:00:00", 100.00)])
    right = pd.DataFrame([row("A", "2026-08-15 10:00:00", 100.00)])
    result = _pair_reconcile(
        left, right, pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-15"),
        "Test PSP", "PSP", "Orchestrator", 0.02,
    )
    assert result["detail"].iloc[0]["Status"] == "Orchestrator Outside Period"
    assert not bool(result["detail"].iloc[0]["Action Required"])


def test_half_open_end_boundary():
    left = pd.DataFrame([row("A", "2026-08-15 05:00:00", 100.00)])
    right = pd.DataFrame([row("A", "2026-08-15 05:00:00", 100.00)])
    result = _pair_reconcile(
        left, right, pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-15 05:00:00"),
        "Test PSP", "PSP", "Orchestrator", 0.02,
    )
    assert result["detail"].empty
