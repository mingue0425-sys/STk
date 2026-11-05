import argparse
import json

import pytest

from stk.cli import main, parse_horizon
from stk.forecast import run_forecast
from stk.plotting import plot_forecast


def test_parse_horizon():
    assert parse_horizon("6m") == 126
    assert parse_horizon("1y") == 252
    assert parse_horizon("2w") == 10
    assert parse_horizon("30d") == 30
    assert parse_horizon("45") == 45
    with pytest.raises(argparse.ArgumentTypeError):
        parse_horizon("abc")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_horizon("0")


def test_cli_scenarios_dump(capsys):
    assert main(["scenarios"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert {s["key"] for s in spec["scenarios"]} >= {"bull", "base", "bear"}


def test_cli_forecast_from_csv(tmp_path, prices, capsys):
    csv = tmp_path / "px.csv"
    prices.to_csv(csv)
    out = tmp_path / "out"
    code = main([
        "SYN", "--csv", str(csv), "--benchmark", "none", "--rf", "0.04", "--no-market",
        "-H", "3m", "-n", "1000", "-o", str(out), "--no-cache",
    ])
    assert code == 0
    text = capsys.readouterr().out
    assert "SYN 시나리오별 주가 흐름 예측" in text
    files = sorted(p.suffix for p in out.iterdir())
    assert files == [".html", ".json", ".md", ".png"]


def test_cli_custom_scenarios(tmp_path, prices, capsys):
    csv = tmp_path / "px.csv"
    prices.to_csv(csv)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "fixed_probabilities": True,
        "scenarios": [
            {"key": "up", "name": "상승", "prior": 0.7, "phases": [{"drift": 0.3}]},
            {"key": "down", "name": "하락", "prior": 0.3, "phases": [{"drift": -0.3, "vol_mult": 1.5}]},
        ],
    }), encoding="utf-8")
    code = main([
        "SYN", "--csv", str(csv), "--benchmark", "none", "--rf", "0.04", "-n", "500",
        "--scenarios", str(spec), "--no-save", "--json", "--no-cache", "--no-market",
    ])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    probs = {s["key"]: s["probability"] for s in data["scenarios"]}
    assert probs == pytest.approx({"up": 0.7, "down": 0.3})


def test_plot_english_fallback(tmp_path, prices):
    fc = run_forecast(prices, "SYN", horizon=63, n_paths=500)
    path = plot_forecast(fc, prices, tmp_path / "c.png", lang="en")
    assert path.exists() and path.stat().st_size > 10_000


def test_parse_event():
    from stk.cli import parse_event

    assert parse_event("2027-01-29") == ("2027-01-29", None)
    assert parse_event("2027-01-29:8%") == ("2027-01-29", pytest.approx(0.08))
    assert parse_event("2027-01-29:0.05") == ("2027-01-29", pytest.approx(0.05))
    with pytest.raises(argparse.ArgumentTypeError):
        parse_event("not-a-date")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_event("2027-01-29:150%")


def test_cli_forecast_writes_html_with_event(tmp_path, prices, capsys):
    csv = tmp_path / "px.csv"
    prices.to_csv(csv)
    out = tmp_path / "out"
    ev = (prices.index[-1] + __import__("pandas").Timedelta(days=20)).strftime("%Y-%m-%d")
    code = main([
        "SYN", "--csv", str(csv), "--benchmark", "none", "--rf", "0.04", "--no-market",
        "-H", "2m", "-n", "500", "-o", str(out), "--no-cache", "--no-plot", "--event", ev,
    ])
    assert code == 0
    assert "이벤트:" in capsys.readouterr().out
    assert sorted(p.suffix for p in out.iterdir()) == [".html", ".json", ".md"]


def test_cli_backtest_compare(tmp_path, prices, capsys):
    csv = tmp_path / "px.csv"
    prices.to_csv(csv)
    code = main([
        "backtest", "SYN", "--csv", str(csv), "--benchmark", "none", "--rf", "0.04", "--no-market",
        "-H", "21", "--step", "126", "--eval-years", "1.5", "-n", "200", "--compare", "--no-save", "--no-cache",
    ])
    assert code == 0
    text = capsys.readouterr().out
    assert "모델 설정 비교" in text and "국면(HMM)" in text


@pytest.mark.parametrize("cmd", [[], ["forecast"], ["backtest"], ["scenarios"], ["watch"], ["track"]])
def test_help_renders(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*cmd, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out
