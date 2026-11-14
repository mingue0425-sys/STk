"""예측 기록과 실전 성과 추적.

예측할 때마다 체크포인트(1주·1개월·3개월…)별 예측 분포(5~95% 분위)를 JSONL 파일에 남기고,
`python -m stk track` 으로 날짜가 지난 체크포인트를 실제 주가와 비교합니다.

수정주가는 배당이 생길 때마다 과거 값 전체가 다시 조정되므로, 실제 수익률은
'현재 시점의 수정주가 시계열'에서 기준일 대비 비율로 계산하고 예측도 수익률(배수)로 비교합니다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .forecast import LOG_QUANTILES, Forecast

DEFAULT_LOG = Path.home() / ".local" / "share" / "stk" / "forecasts.jsonl"


def log_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("STK_LOG")
    return Path(env) if env else DEFAULT_LOG


def forecast_record(fc: Forecast) -> dict:
    as_of = fc.stats.last_date.strftime("%Y-%m-%d")
    return {
        "id": f"{fc.symbol}|{as_of}|{fc.horizon}|{fc.model}",
        "symbol": fc.symbol,
        "as_of": as_of,
        "created": datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "price": fc.s0,
        "horizon": fc.horizon,
        "model": fc.model,
        "regime_weight": fc.regime_weight,
        "trend": fc.trend_tilt,
        "prob_up": fc.mix_prob_up,
        "expected_return": fc.mix_expected_price / fc.s0 - 1,
        "outlook": fc.outlook.label,
        "direction": fc.outlook.direction,
        "regime": (fc.regime.state.names[fc.regime.state.current_regime] if fc.regime is not None else None),
        "regime_prob": (float(fc.regime.state.current.max()) if fc.regime is not None else None),
        "var95": fc.risk.var95 if fc.risk is not None else None,
        "quantile_levels": list(LOG_QUANTILES),
        "checkpoints": [
            {"label": lbl, "day": d, "date": fc.dates[d].strftime("%Y-%m-%d"), "q": fc.cp_quantiles.get(d, [])}
            for lbl, d in fc.checkpoints
        ],
    }


def append_log(fc: Forecast, path: str | Path | None = None) -> Path:
    """기록을 추가합니다. 같은 종목·기준일·기간·모델 기록이 있으면 교체합니다."""
    p = log_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = forecast_record(fc)
    records = [r for r in load_log(p) if r.get("id") != rec["id"]]
    records.append(rec)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    tmp.replace(p)
    return p


def load_log(path: str | Path | None = None) -> list[dict]:
    p = log_path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _pit(levels: list[float], qs_ret: np.ndarray, realized: float) -> float:
    """5~95% 분위로 근사한 예측 CDF 에서 실제값의 위치 (구간 밖은 2.5%/97.5%로 둠)."""
    if realized <= qs_ret[0]:
        return 0.025
    if realized >= qs_ret[-1]:
        return 0.975
    return float(np.interp(realized, qs_ret, levels))


@dataclass
class TrackResult:
    evaluations: pd.DataFrame   # 체크포인트 평가 행
    pending: pd.DataFrame       # 아직 날짜가 오지 않은 기록
    summary: dict


def evaluate(records: list[dict], price_loader) -> TrackResult:
    """price_loader(symbol) → 수정주가 DataFrame/Series (현재 시점 기준)."""
    rows, pending = [], []
    cache: dict[str, pd.Series | None] = {}
    for rec in records:
        sym = rec["symbol"]
        if sym not in cache:
            try:
                px = price_loader(sym)
                cache[sym] = (px["adj_close"] if hasattr(px, "columns") else px).astype(float).dropna()
            except Exception:  # noqa: BLE001 - 데이터 실패는 해당 종목만 건너뜀
                cache[sym] = None
        series = cache[sym]
        if series is None or series.empty:
            continue
        as_of = pd.Timestamp(rec["as_of"])
        base = series.loc[:as_of]
        if base.empty:
            continue
        base_price = float(base.iloc[-1])
        last_date = series.index[-1]
        levels = rec.get("quantile_levels", list(LOG_QUANTILES))
        for cp in rec["checkpoints"]:
            cp_date = pd.Timestamp(cp["date"])
            if cp_date > last_date or not cp.get("q"):
                pending.append({"symbol": sym, "as_of": rec["as_of"], "horizon": rec["horizon"],
                                "checkpoint": cp["label"], "date": cp["date"]})
                continue
            realized = float(series.loc[:cp_date].iloc[-1]) / base_price  # 실제 수익 배수
            q = np.asarray(cp["q"], dtype=float) / float(rec["price"])     # 예측 수익 배수
            lv = np.asarray(levels)
            i05, i25, i75, i95 = (int(np.argmin(np.abs(lv - x))) for x in (0.05, 0.25, 0.75, 0.95))
            i50 = int(np.argmin(np.abs(lv - 0.5)))
            final = cp["day"] == rec["horizon"]
            rows.append({
                "symbol": sym,
                "as_of": rec["as_of"],
                "horizon": rec["horizon"],
                "model": rec.get("model", ""),
                "checkpoint": cp["label"],
                "date": cp["date"],
                "final": final,
                "realized_return": realized - 1,
                "median_return": q[i50] - 1,
                "in50": bool(q[i25] <= realized <= q[i75]),
                "in90": bool(q[i05] <= realized <= q[i95]),
                "pit": _pit(levels, q, realized),
                "prob_up": rec.get("prob_up") if final else None,
                "hit": (bool((rec["prob_up"] > 0.5) == (realized > 1)) if final and rec.get("prob_up") != 0.5 else None),
            })
    ev = pd.DataFrame(rows)
    pend = pd.DataFrame(pending)
    summary = {"records": len(records), "evaluated": int(len(ev)), "pending": int(len(pend))}
    if len(ev):
        finals = ev[ev["final"]]
        summary.update({
            "coverage50": float(ev["in50"].mean()),
            "coverage90": float(ev["in90"].mean()),
            "mean_pit": float(ev["pit"].mean()),
            "final_n": int(len(finals)),
            "hit_rate": float(finals["hit"].dropna().mean()) if finals["hit"].notna().any() else float("nan"),
        })
    return TrackResult(ev, pend, summary)


def render_track(result: TrackResult, log_file: Path) -> str:
    from .report import _indent, pct, table

    s = result.summary
    out = ["═" * 78, "  예측 기록 실전 성과 추적", "═" * 78, f"기록 파일: {log_file}"]
    out.append(f"기록 {s['records']}건 · 평가된 체크포인트 {s['evaluated']}개 · 대기 중 {s['pending']}개")
    if not s["evaluated"]:
        out.append("")
        out.append("아직 날짜가 지난 체크포인트가 없습니다. 예측 후 1주 이상 지나면 평가할 수 있습니다.")
        return "\n".join(out)
    out.append("")
    rows = [
        ["50% 구간 적중률", pct(s["coverage50"], False), "50%"],
        ["90% 구간 적중률", pct(s["coverage90"], False), "90%"],
        ["평균 PIT", f"{s['mean_pit']:.2f}", "0.50"],
        ["방향 적중률 (만기 도달분)", pct(s["hit_rate"], False) if s["final_n"] else "-", f"{s['final_n']}건"],
    ]
    out.append(_indent(table(["지표", "결과", "이상적 값 / 표본"], rows)))
    out.append("")
    ev = result.evaluations.sort_values(["as_of", "symbol", "horizon", "date"]).tail(30)
    rows = [
        [r.symbol, r.as_of, f"{r.horizon}일", r.checkpoint, r.date, pct(r.realized_return), pct(r.median_return),
         "●" if r.in50 else "○", "●" if r.in90 else "○", f"{r.pit:.2f}"]
        for r in ev.itertuples()
    ]
    out.append("  최근 평가 (최대 30개, ● = 구간 안)")
    out.append(_indent(table(["종목", "기준일", "예측 기간", "체크포인트", "날짜", "실제 수익률", "예측 중앙값", "50%", "90%", "PIT"], rows,
                             ["left", "left", "right", "left", "left", "right", "right", "center", "center", "right"])))
    if s["evaluated"] < 30:
        out.append("")
        out.append("  ※ 표본이 적어 아직 통계적으로 의미 있는 평가는 아닙니다. 꾸준히 예측을 기록하면 신뢰도가 쌓입니다.")
    return "\n".join(out)
