"""예측 결과를 사람이 읽기 쉬운 한국어 해설로 바꿉니다 (규칙 기반, 숫자는 모두 예측 결과에서 가져옴)."""

from __future__ import annotations

import math

from .forecast import Forecast, horizon_label


def _p(x: float, signed: bool = False) -> str:
    if signed:
        return f"{x:+.1%}" if abs(x) < 0.1 else f"{x:+.0%}"
    return f"{x:.0%}"


def _money(x: float) -> str:
    return f"${x:,.2f}" if x < 10_000 else f"${x:,.0f}"


def narrative(fc: Forecast) -> list[str]:
    """3~8개 문장으로 된 요약 해설."""
    ol, st = fc.outlook, fc.stats
    h = horizon_label(fc.horizon)
    out: list[str] = []

    # 1. 전체 흐름
    first = (f"{fc.symbol}의 향후 {h} 흐름은 '{ol.label}'으로 보입니다. 상승 마감 확률 {_p(ol.prob_up)}, "
             f"기대수익률 {_p(ol.expected_return, True)}이며 {ol.shape.replace('경로', '흐름')}이 중심 시나리오입니다.")
    out.append(first)
    if ol.direction == 0:
        out.append("오르내릴 방향보다는 움직일 수 있는 폭에 무게를 두고 보는 것이 좋습니다.")

    # 2. 범위·위험
    lo, hi = ol.band90
    sent = f"{h} 뒤 주가는 90% 확률로 {_money(lo)}~{_money(hi)} 사이로 예상되며, 변동성 수준은 '{ol.volatility.split(' (')[0]}'입니다."
    if fc.risk is not None:
        sent += f" 나쁜 5% 경우의 평균 손실은 약 {fc.risk.cvar95:.0%}입니다."
    out.append(sent)

    # 3. 국면
    if fc.regime is not None:
        rs = fc.regime.state
        cur = rs.current_regime
        later = rs.occupancy(fc.checkpoints[-1][1])
        txt = f"현재 '{rs.names[cur]}' 국면일 가능성이 {_p(rs.current[cur])}"
        if cur == len(rs.names) - 1 and rs.current[cur] >= 0.5:
            txt += "로 변동성이 큰 시기라 단기 급등락에 유의해야 하며,"
        elif cur == 0 and rs.current[cur] >= 0.5:
            txt += "로 비교적 안정적인 시기이며,"
        else:
            txt += "이며,"
        txt += f" {h} 뒤에는 안정 국면 확률 {_p(later[0])}, 불안 국면 확률 {_p(later[-1])}로 예상됩니다."
        out.append(txt)

    # 4. 시장 환경
    m = fc.market
    if m is not None and m.vix is not None:
        txt = f"시장 전체는 '{m.label}' 상태(VIX {m.vix:.1f})"
        if m.stress >= 0.15:
            txt += "로, 급락·조정 시나리오의 확률을 평소보다 높게 반영했습니다."
        elif m.stress <= -0.4:
            txt += "로 안정적이어서 급락 시나리오 비중이 낮게 반영됐습니다."
        else:
            txt += "입니다."
        out.append(txt)

    # 5. 실적·이벤트
    auto = [e for e in fc.events if e.source == "auto"]
    user = [e for e in fc.events if e.source != "auto"]
    if not auto and fc.next_earnings and fc.earnings is not None:
        d = fc.next_earnings[0]
        out.append(f"다음 실적 발표는 {d:%m월 %d일} 전후(±{fc.earnings.spread_days}거래일)로 추정됩니다. 과거 실적 반응일의 "
                   f"하루 변동이 ±{fc.earnings.event_std:.0%} 안팎이었으니 그 무렵 급등락에 유의하세요.")
    if auto:
        e = auto[0]
        out.append(f"{e.date:%m월 %d일} 전후(±{e.spread}거래일)로 실적 발표가 예상되어(과거 패턴으로 추정), "
                   f"그 무렵 하루 ±{e.std:.0%} 안팎의 급등락 가능성을 반영했습니다.")
    if user:
        out.append("지정한 이벤트일: " + ", ".join(f"{e.date:%m월 %d일}(±{e.std:.0%})" for e in user) + ".")

    # 6. 섹터
    sec = fc.sector
    if sec is not None:
        if sec.method == "raw":
            txt = (f"특정 섹터와의 연관은 약하고(가장 가까운 섹터: {sec.name}), 주가 변동의 "
                   f"{_p(sec.share_market)}는 시장, {_p(sec.share_idio)}는 종목 고유 요인에서 나옵니다.")
        else:
            strength = "강하게" if sec.fit >= 0.6 else "어느 정도" if sec.fit >= 0.4 else "약하게"
            txt = (f"{sec.name} 섹터({sec.etf})와 {strength} 함께 움직이며, 주가 변동은 시장 {_p(sec.share_market)} · "
                   f"섹터 {_p(sec.share_sector)} · 종목 고유 {_p(sec.share_idio)}로 나뉩니다.")
        if math.isfinite(sec.rel_3m):
            txt += f" 최근 3개월은 섹터보다 {abs(sec.rel_3m):.0%}p {'강했습니다' if sec.rel_3m > 0 else '약했습니다'}."
        out.append(txt)

    # 7. 추세 (확률 반영 여부)
    if abs(st.trend_score) >= 0.3 and not fc.trend_tilt:
        out.append(f"최근 추세는 {st.trend_label}(점수 {st.trend_score:+.2f})이지만, 백테스트상 추세가 방향 예측에 "
                   "도움이 되지 않아 확률에는 반영하지 않았습니다.")

    # 8. 목표가·손절가
    if fc.race:
        r = fc.race
        out.append(f"목표가 {_money(r['target'])}에 먼저 닿을 확률은 {_p(r['target_first'])}, "
                   f"손절가 {_money(r['stop'])}에 먼저 닿을 확률은 {_p(r['stop_first'])}입니다.")

    if fc.data_warnings:
        out.append("데이터 점검에서 확인이 필요한 부분이 있습니다: " + fc.data_warnings[0])
    return out


def narrative_text(fc: Forecast) -> str:
    return " ".join(narrative(fc))
