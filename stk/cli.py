"""명령행 인터페이스.

    python -m stk AAPL                       # 6개월 시나리오 예측 (forecast 생략 가능)
    python -m stk forecast NVDA TSLA -H 1y   # 여러 종목, 1년 예측 + 비교표
    python -m stk backtest AAPL -H 3m        # 워크포워드 백테스트
    python -m stk backtest AAPL --compare    # 모델 설정별 백테스트 비교
    python -m stk AAPL --event 2027-01-29    # 실적 발표일 충격 반영
    python -m stk scenarios > my.json        # 기본 시나리오 설정 출력 (수정해서 --scenarios 로 사용)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import __version__
from .data import (
    DEFAULT_CACHE_DIR,
    DataError,
    get_history,
    get_risk_free_series,
    load_csv,
    normalize_symbol,
    risk_free_at,
)
from .forecast import DEFAULT_MODEL, DEFAULT_REGIME_WEIGHT, DEFAULT_TREND_ON, MODELS
from .scenarios import DEFAULT_ERP, DEFAULT_TREND_WEIGHT, TREND_WEIGHT_ON, load_spec, spec_to_json

SUBCOMMANDS = {"forecast", "backtest", "scenarios"}
VIX_SYMBOL, VIX3M_SYMBOL = "^VIX", "^VIX3M"


def parse_horizon(text: str) -> int:
    """'6m', '1y', '2w', '30d', '90' → 거래일 수."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([dwmy]?)\s*", str(text).lower())
    if not m:
        raise argparse.ArgumentTypeError(f"기간 형식 오류: {text!r} (예: 30d, 2w, 6m, 1y)")
    value, unit = float(m.group(1)), m.group(2) or "d"
    days = int(round(value * {"d": 1, "w": 5, "m": 21, "y": 252}[unit]))
    if not 1 <= days <= 2520:
        raise argparse.ArgumentTypeError("예측 기간은 1거래일 ~ 10년 사이여야 합니다")
    return days


def parse_event(text: str) -> tuple[str, float | None]:
    """'2027-01-29' 또는 '2027-01-29:0.08' / '2027-01-29:8%' → (날짜, 충격 표준편차)."""
    date, _, size = str(text).partition(":")
    try:
        import pandas as pd

        pd.Timestamp(date)
    except (ValueError, TypeError) as err:
        raise argparse.ArgumentTypeError(f"이벤트 날짜 형식 오류: {text!r} (예: 2027-01-29 또는 2027-01-29:8%)") from err
    if not size:
        return date, None
    size = size.strip()
    try:
        value = float(size[:-1]) / 100 if size.endswith("%") else float(size)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"이벤트 크기 형식 오류: {size!r}") from err
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("이벤트 충격 크기는 0~100% 사이여야 합니다")
    return date, value


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--years", type=float, default=10, help="내려받을 과거 데이터 기간(년), 기본 10")
    p.add_argument("--benchmark", default="SPY", help="베타 계산용 벤치마크, 기본 SPY ('none'이면 생략)")
    p.add_argument("--rf", default="auto", help="무위험이자율(예: 0.042). 기본 auto = 13주 T-bill(^IRX)")
    p.add_argument("--erp", type=float, default=DEFAULT_ERP, help=f"주식 위험 프리미엄, 기본 {DEFAULT_ERP}")
    p.add_argument(
        "--trend", action=argparse.BooleanOptionalAction, default=DEFAULT_TREND_ON,
        help=f"추세 신호 사용: 추세 점수로 시나리오 확률 조정 + 기본 기대수익률에 추세 {TREND_WEIGHT_ON * 100:.0f}%% 반영 "
             f"(기본 {'켬' if DEFAULT_TREND_ON else '끔'})",
    )
    p.add_argument(
        "--trend-weight", type=float, default=None,
        help=f"기본 기대수익률 중 최근 추세 비중(0~1). 생략 시 --trend 면 {TREND_WEIGHT_ON}, 아니면 {DEFAULT_TREND_WEIGHT}",
    )
    p.add_argument("--no-stress", action="store_true", help="시장 스트레스로 시나리오 확률을 조정하지 않음")
    p.add_argument("--scenarios", metavar="JSON", help="사용자 시나리오 설정 파일")
    p.add_argument("--no-tilt", action="store_true", help=argparse.SUPPRESS)  # 구버전 호환: --no-trend 와 동일
    p.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                   help=f"예측 모델: ensemble(시나리오+국면), scenario, regime (기본 {DEFAULT_MODEL})")
    p.add_argument("--regime-weight", type=float, default=DEFAULT_REGIME_WEIGHT,
                   help=f"ensemble 에서 국면(HMM) 모델 비중, 기본 {DEFAULT_REGIME_WEIGHT}")
    p.add_argument("--no-market", action="store_true", help="VIX·시장 환경을 반영하지 않음")
    p.add_argument("--csv", metavar="FILE", help="네트워크 대신 로컬 CSV 사용 (종목 1개일 때)")
    p.add_argument("--refresh", action="store_true", help="캐시를 무시하고 새로 다운로드")
    p.add_argument("--no-cache", action="store_true", help="로컬 캐시를 사용하지 않음")
    p.add_argument("--seed", type=int, default=42, help="난수 시드 (재현성), 기본 42")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stk",
        description="미국 개별주 시나리오별 주가 흐름 예측 (몬테카를로 시뮬레이션)",
    )
    parser.add_argument("--version", action="version", version=f"stk {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("forecast", help="시나리오별 주가 흐름 예측")
    f.add_argument("tickers", nargs="+", help="미국 주식 티커 (예: AAPL MSFT NVDA)")
    f.add_argument("-H", "--horizon", type=parse_horizon, default=parse_horizon("6m"),
                   help="예측 기간: 30d, 2w, 3m, 6m, 1y ... (기본 6m = 126거래일)")
    f.add_argument("-n", "--sims", type=int, default=10_000, help="시나리오당 시뮬레이션 경로 수, 기본 10000")
    f.add_argument("-o", "--out", default="output", help="결과 저장 폴더 (PNG/MD/JSON), 기본 ./output")
    f.add_argument("--no-save", action="store_true", help="파일로 저장하지 않고 화면에만 출력")
    f.add_argument("--no-plot", action="store_true", help="차트를 만들지 않음")
    f.add_argument("--lang", choices=["auto", "ko", "en"], default="auto", help="차트 라벨 언어 (기본: 한글 글꼴 있으면 ko)")
    f.add_argument("--json", action="store_true", help="텍스트 리포트 대신 JSON 요약을 출력")
    f.add_argument("--event", type=parse_event, action="append", metavar="DATE[:SIZE]",
                   help="실적 발표 등 이벤트일 (여러 번 지정 가능). SIZE 생략 시 과거 데이터로 추정")
    _add_common(f)

    b = sub.add_parser("backtest", help="과거 시점별 예측 정확도(보정) 평가")
    b.add_argument("ticker")
    b.add_argument("-H", "--horizon", type=parse_horizon, default=parse_horizon("3m"), help="예측 기간 (기본 3m)")
    b.add_argument("--step", type=int, default=21, help="예측 시점 간격(거래일), 기본 21")
    b.add_argument("--eval-years", type=float, default=5, help="평가 기간(최근 N년), 기본 5")
    b.add_argument("-n", "--sims", type=int, default=2000, help="시나리오당 경로 수, 기본 2000")
    b.add_argument("-o", "--out", default="output", help="결과 CSV 저장 폴더")
    b.add_argument("--no-save", action="store_true")
    b.add_argument("--compare", action="store_true", help="모델 설정별(앙상블/시나리오/국면/추세 제외) 성능 비교")
    _add_common(b)

    s = sub.add_parser("scenarios", help="기본 시나리오 설정(JSON) 출력")
    s.add_argument("-o", "--output", help="파일로 저장")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 출력
        except (AttributeError, ValueError):
            pass

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "forecast")
    args = build_parser().parse_args(argv)

    if args.command == "scenarios":
        text = spec_to_json()
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
            print(f"저장: {args.output}")
        else:
            print(text)
        return 0
    if args.command == "backtest":
        return _cmd_backtest(args)
    return _cmd_forecast(args)


# ------------------------------------------------------------------ 공통 준비
def _context(args) -> dict:
    cache_dir = None if args.no_cache else DEFAULT_CACHE_DIR
    bench = None
    if args.benchmark and args.benchmark.lower() != "none":
        try:
            bench = get_history(args.benchmark, years=args.years, cache_dir=cache_dir, refresh=args.refresh)
        except DataError as err:
            print(f"[경고] 벤치마크 데이터를 가져오지 못했습니다: {err} → 베타 1.0 가정", file=sys.stderr)
    rf_series = None
    rf_fixed = None
    if str(args.rf).lower() == "auto":
        rf_series = get_risk_free_series(years=args.years, cache_dir=cache_dir, refresh=args.refresh)
        if rf_series is None:
            print("[경고] 무위험이자율(^IRX)을 가져오지 못해 4.0%를 사용합니다", file=sys.stderr)
    else:
        rf_fixed = float(args.rf)
    vix = vix3m = None
    if not args.no_market and bench is not None:  # 시장 환경은 S&P500(벤치마크) 데이터가 있어야 계산
        for sym in (VIX_SYMBOL, VIX3M_SYMBOL):
            try:
                frame = get_history(sym, years=args.years, cache_dir=cache_dir, refresh=args.refresh)
            except DataError:
                frame = None
            if sym == VIX_SYMBOL:
                vix = frame
            else:
                vix3m = frame
        if vix is None:
            print("[경고] VIX 데이터를 가져오지 못해 시장 내재변동성을 반영하지 않습니다", file=sys.stderr)
    spec = load_spec(args.scenarios)
    return {
        "cache_dir": cache_dir, "bench": bench, "rf_series": rf_series, "rf_fixed": rf_fixed, "spec": spec,
        "vix": vix, "vix3m": vix3m,
    }


def _trend_settings(args) -> tuple[bool, float]:
    trend_on = bool(args.trend) and not args.no_tilt
    weight = args.trend_weight if args.trend_weight is not None else (TREND_WEIGHT_ON if trend_on else DEFAULT_TREND_WEIGHT)
    return trend_on, weight


def _load_prices(args, ticker: str, cache_dir):
    if args.csv:
        return load_csv(args.csv)
    return get_history(ticker, years=args.years, cache_dir=cache_dir, refresh=args.refresh)


# ------------------------------------------------------------------ forecast
def _cmd_forecast(args) -> int:
    from .forecast import run_forecast
    from .html_report import render_html
    from .market import market_context
    from .report import render_comparison, render_markdown, render_text, write_json

    if args.csv and len(args.tickers) > 1:
        print("--csv 는 종목 1개일 때만 사용할 수 있습니다", file=sys.stderr)
        return 2
    try:
        ctx = _context(args)
    except (OSError, ValueError) as err:
        print(f"[오류] {err}", file=sys.stderr)
        return 2

    market = None
    if not args.no_market and ctx["bench"] is not None:
        market = market_context(ctx["bench"], ctx["vix"], ctx["vix3m"])

    trend_on, trend_weight = _trend_settings(args)
    forecasts = []
    for raw in args.tickers:
        ticker = normalize_symbol(raw)
        try:
            prices = _load_prices(args, ticker, ctx["cache_dir"])
            rf = ctx["rf_fixed"] if ctx["rf_fixed"] is not None else risk_free_at(ctx["rf_series"])
            fc = run_forecast(
                prices,
                ticker,
                horizon=args.horizon,
                n_paths=args.sims,
                benchmark=ctx["bench"],
                risk_free=rf,
                erp=args.erp,
                trend_weight=trend_weight,
                spec=ctx["spec"],
                tilt=trend_on,
                stress_tilt=not args.no_stress,
                seed=args.seed,
                market=market,
                model=args.model,
                regime_weight=args.regime_weight,
                events=args.event,
            )
        except (DataError, ValueError) as err:
            print(_error_text(ticker, err), file=sys.stderr)
            continue
        forecasts.append(fc)

        if args.json:
            print(json.dumps(_brief(fc), ensure_ascii=False, indent=2))
        else:
            print(render_text(fc))

        if not args.no_save:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            stem = f"{ticker}_{fc.stats.last_date:%Y%m%d}_{args.horizon}d"
            chart_name = None
            if not args.no_plot:
                from .plotting import plot_forecast

                lang = None if args.lang == "auto" else args.lang
                chart = plot_forecast(fc, prices, out / f"{stem}.png", lang=lang)
                chart_name = chart.name
            (out / f"{stem}.md").write_text(render_markdown(fc, chart_name), encoding="utf-8")
            (out / f"{stem}.html").write_text(render_html(fc, prices), encoding="utf-8")
            write_json(fc, out / f"{stem}.json")
            if not args.json:
                exts = (["png"] if chart_name else []) + ["html", "md", "json"]
                print(f"\n저장됨: {', '.join(str(out / f'{stem}.{ext}') for ext in exts)}")
        if not args.json:
            print()

    if len(forecasts) > 1 and not args.json:
        print(render_comparison(forecasts))
    return 0 if forecasts else 1


def _error_text(ticker: str, err: Exception) -> str:
    msg = str(err)
    return f"[오류] {msg}" if msg.startswith(ticker) else f"[오류] {ticker}: {msg}"


def _brief(fc) -> dict:
    d = fc.to_dict()
    d.pop("path", None)
    return d


# ------------------------------------------------------------------ backtest
def _cmd_backtest(args) -> int:
    from .backtest import compare_configs, render_backtest, render_comparison_table, walk_forward

    ticker = normalize_symbol(args.ticker)
    try:
        ctx = _context(args)
        prices = _load_prices(args, ticker, ctx["cache_dir"])
        rf_series = ctx["rf_series"]
        if ctx["rf_fixed"] is not None:
            import pandas as pd

            rf_series = pd.Series([ctx["rf_fixed"]], index=[prices.index[0]])
        common = dict(
            horizon=args.horizon,
            step=args.step,
            years=args.eval_years,
            n_paths=args.sims,
            benchmark=ctx["bench"],
            risk_free_series=rf_series,
            vix=ctx["vix"],
            vix3m=ctx["vix3m"],
            use_market=not args.no_market,
            progress=sys.stdout.isatty(),
            erp=args.erp,
            spec=ctx["spec"],
        )
        if args.compare:
            table = compare_configs(prices, ticker, regime_weight=args.regime_weight, **common)
            print(render_comparison_table(ticker, args.horizon, table))
            if not args.no_save:
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{ticker}_backtest_compare_{args.horizon}d.csv"
                table.to_csv(path, index=False)
                print(f"\n저장됨: {path}")
            return 0
        trend_on, trend_weight = _trend_settings(args)
        bt = walk_forward(
            prices,
            ticker,
            trend_weight=trend_weight,
            tilt=trend_on,
            stress_tilt=not args.no_stress,
            model=args.model,
            regime_weight=args.regime_weight,
            **common,
        )
    except (DataError, ValueError, OSError) as err:
        print(_error_text(ticker, err), file=sys.stderr)
        return 1
    print(render_backtest(bt))
    if not args.no_save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{ticker}_backtest_{args.horizon}d.csv"
        bt.records.to_csv(path, index=False)
        print(f"\n저장됨: {path}")
    return 0
