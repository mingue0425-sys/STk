"""예측 차트 (matplotlib, PNG).

- 위: 최근 1년 가격 + 확률가중 팬 차트(5~95%, 25~75%) + 시나리오별 중앙값 경로
- 아래: 만기 수익률 분포 (시나리오별, 확률 가중)

한글 글꼴(맑은 고딕, AppleGothic, 나눔고딕, Noto Sans CJK 등)이 있으면 한글로,
없으면 영어 라벨로 그립니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .forecast import Forecast

# 검증된 범주형 팔레트(라이트 모드) — 시나리오 순서대로 고정 배정
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BAND = "#8a8984"

KOREAN_FONTS = [
    "Malgun Gothic",
    "AppleGothic",
    "Apple SD Gothic Neo",
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Source Han Sans KR",
    "UnDotum",
]

EN_NAMES = {
    "bull": "Bull",
    "base": "Base",
    "bear": "Bear",
    "dip_recovery": "Dip & recovery",
    "crash": "Crash",
}

LABELS = {
    "ko": {
        "history": "실제 주가",
        "median": "전체 중앙값",
        "band90": "전체 90% 구간",
        "band50": "전체 50% 구간",
        "price": "주가 (USD)",
        "dist": "만기 수익률 분포 (확률 가중)",
        "ret": "만기 수익률",
        "density": "상대 빈도",
        "total": "전체",
        "regime": "국면 모델",
        "weight": "비중",
        "ref": "참고",
        "event": "이벤트",
        "today": "기준일",
        "subtitle": "기준일 {date} · 현재가 ${price:,.2f} · {h}거래일 예측 · 상승확률 {pu:.0%} · 기대수익률 {er:+.1%}",
    },
    "en": {
        "history": "Price",
        "median": "Overall median",
        "band90": "Overall 90% band",
        "band50": "Overall 50% band",
        "price": "Price (USD)",
        "dist": "Terminal return distribution (probability-weighted)",
        "ret": "Return at horizon",
        "density": "Relative frequency",
        "total": "Overall",
        "regime": "Regime model",
        "weight": "weight",
        "ref": "ref.",
        "event": "Event",
        "today": "As of",
        "subtitle": "As of {date} · Last ${price:,.2f} · {h}-day horizon · P(up) {pu:.0%} · E[return] {er:+.1%}",
    },
}

OUTLOOK_EN = {
    2: "Clear uptrend",
    1: "Mild uptrend",
    0: "Sideways / no clear direction",
    -1: "Mild downtrend",
    -2: "Clear downtrend",
}


def _setup_font() -> str:
    """사용 가능한 한글 글꼴을 찾아 설정하고 언어 코드('ko'/'en')를 반환합니다."""
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return "ko"
    return "en"


def _spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """세로 위치가 겹치지 않도록 라벨 y좌표를 벌립니다."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        if placed[i] - placed[i - 1] < min_gap:
            placed[i] = placed[i - 1] + min_gap
    # 위로 밀린 만큼 전체를 가운데로 재조정
    shift = (placed - np.array(ys)[order]).mean()
    placed -= shift / 2
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def plot_forecast(
    fc: Forecast,
    prices,
    path: str | Path,
    history_days: int = 252,
    lang: str | None = None,
    dpi: int = 150,
) -> Path:
    import logging

    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    detected = _setup_font()
    lang = "en" if (lang == "ko" and detected != "ko") else (lang or detected)
    L = LABELS[lang]

    def sc_name(sc) -> str:
        if lang == "ko":
            return sc.name
        return EN_NAMES.get(sc.key, sc.key)

    series = prices["adj_close"] if hasattr(prices, "columns") else prices
    hist = series.dropna().iloc[-history_days:]
    s0 = fc.s0
    dates = fc.dates

    fig = plt.figure(figsize=(12, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        a.grid(True, color=GRID, linewidth=0.7)
        a.set_axisbelow(True)

    # ---- 상단: 팬 차트
    mq = fc.mix_quantiles
    ax.fill_between(dates, mq[0.05], mq[0.95], color=BAND, alpha=0.16, linewidth=0, label=L["band90"])
    ax.fill_between(dates, mq[0.25], mq[0.75], color=BAND, alpha=0.30, linewidth=0, label=L["band50"])
    ax.plot(hist.index, hist.to_numpy(), color=TEXT_PRIMARY, linewidth=1.6, label=L["history"])

    end_points = []
    for res in fc.results:
        color = SERIES_COLORS[res.scenario.color_slot % len(SERIES_COLORS)]
        med = res.quantiles[0.5]
        ax.plot(dates, med, color=color, linewidth=2.0, label=f"{sc_name(res.scenario)} ({res.scenario.probability:.0%})")
        end_points.append((float(med[-1]), color, sc_name(res.scenario)))
    regime_color = SERIES_COLORS[len(fc.results) % len(SERIES_COLORS)]
    if fc.regime is not None:
        med = fc.regime.quantiles[0.5]
        wtxt = f"{L['weight']} {fc.regime_weight:.0%}" if fc.regime_weight > 0 else L["ref"]
        ax.plot(dates, med, color=regime_color, linewidth=2.0, linestyle=(0, (1, 1.5)), label=f"{L['regime']} ({wtxt})")
        end_points.append((float(med[-1]), regime_color, L["regime"]))
    ax.plot(dates, mq[0.5], color=TEXT_PRIMARY, linewidth=2.0, linestyle=(0, (4, 2)), label=L["median"])
    ax.axvline(dates[0], color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    for ev_date, _ in fc.events:
        ax.axvline(ev_date, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.annotate(L["event"], xy=(ev_date, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10),
                    textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY)

    # 오른쪽 여백에 직접 라벨 (x: 축 비율, y: 가격)
    x_start = hist.index[0] if len(hist) else dates[0]
    ax.set_xlim(x_start, dates[-1] + (dates[-1] - x_start) * 0.01)
    ymin, ymax = ax.get_ylim()
    ys = _spread_labels([p for p, _, _ in end_points], (ymax - ymin) * 0.045)
    label_tf = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for (y_true, color, name), y_lab in zip(end_points, ys):
        ax.plot([dates[-1]], [y_true], marker="o", markersize=5, color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, clip_on=False, zorder=5)
        ax.annotate(
            f"{name}  ${y_true:,.0f} ({y_true / s0 - 1:+.0%})",
            xy=(dates[-1], y_true),
            xytext=(1.025, y_lab),
            textcoords=label_tf,
            annotation_clip=False,
            fontsize=8.5,
            color=TEXT_PRIMARY,
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, linewidth=1.0, shrinkA=0, shrinkB=3),
        )
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    span_months = (dates[-1] - x_start).days / 30.4
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, int(round(span_months / 12)))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel(L["price"], color=TEXT_SECONDARY, fontsize=9)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)

    outlook_txt = fc.outlook.label if lang == "ko" else OUTLOOK_EN[fc.outlook.direction]
    fig.suptitle(f"{fc.symbol} — {outlook_txt}", x=0.07, ha="left", fontsize=15, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_title(
        L["subtitle"].format(
            date=fc.stats.last_date.strftime("%Y-%m-%d"),
            price=s0,
            h=fc.horizon,
            pu=fc.mix_prob_up,
            er=fc.mix_expected_price / s0 - 1,
        ),
        loc="left",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        pad=10,
    )

    # ---- 하단: 만기 수익률 분포
    groups = [(r.terminal, r.scenario.probability * (1 - fc.regime_weight),
               SERIES_COLORS[r.scenario.color_slot % len(SERIES_COLORS)], sc_name(r.scenario), "-")
              for r in fc.results]
    if fc.regime is not None and fc.regime_weight > 0:
        groups.append((fc.regime.terminal, fc.regime_weight, regime_color, L["regime"], (0, (1, 1.5))))
    all_ret = np.concatenate([g[0] / s0 - 1 for g in groups])
    lo, hi = np.quantile(all_ret, [0.005, 0.995])
    bins = np.linspace(lo, hi, 70)
    width = bins[1] - bins[0]
    total = np.zeros(len(bins) - 1)
    for terminal, weight, color, name, style in groups:
        h, _ = np.histogram(terminal / s0 - 1, bins=bins)
        h = h / len(terminal) * weight / width
        total += h
        ax2.step(bins[:-1], h, where="post", color=color, linewidth=1.6, linestyle=style, label=name)
    ax2.fill_between(bins[:-1], total, step="post", color=BAND, alpha=0.22, linewidth=0, label=L["total"])
    ax2.axvline(0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel(L["ret"], color=TEXT_SECONDARY, fontsize=9)
    ax2.set_title(L["dist"], loc="left", fontsize=10, color=TEXT_PRIMARY)
    leg2 = ax2.legend(loc="upper right", fontsize=8, frameon=False)
    for t in leg2.get_texts():
        t.set_color(TEXT_PRIMARY)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
