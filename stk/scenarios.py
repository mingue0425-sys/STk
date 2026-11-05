"""시나리오 정의, 파라미터 보정(calibration), 확률 배정.

시나리오는 하나 이상의 '국면(phase)'으로 구성됩니다. 각 국면은 예측 기간 중
차지하는 비중(frac)과 그 기간의 기대수익률·변동성 배수·점프(급등락) 특성을 가집니다.
국면을 이어 붙여 "조정 후 반등" 같은 흐름(경로 모양)을 표현할 수 있습니다.

기대수익률은 두 가지 방식으로 지정할 수 있습니다.
- ``drift``         : 절대값 (연율, 예: 0.12 = 연 12%)
- ``drift_spread``  : 기본 기대수익률 + k × 시나리오 폭(spread). 종목별로 자동 보정됩니다.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .analysis import MarketStats

DEFAULT_ERP = 0.05            # 주식 위험 프리미엄 (연)
DEFAULT_TREND_WEIGHT = 0.20   # 기본 기대수익률에서 최근 추세가 차지하는 비중
DEFAULT_VOL_HALFLIFE = 42     # 현재 변동성이 장기 변동성으로 회귀하는 반감기 (거래일)
DEFAULT_TILT_STRENGTH = 0.7   # 추세 점수가 시나리오 확률을 기울이는 강도

DEFAULT_SPEC: dict = {
    "tilt_strength": DEFAULT_TILT_STRENGTH,
    "fixed_probabilities": False,
    "scenarios": [
        {
            "key": "bull",
            "name": "강세 지속",
            "description": "실적·수급 호조로 상승 흐름이 이어지는 경우",
            "prior": 0.25,
            "tilt": 1.0,
            "phases": [{"frac": 1.0, "drift_spread": 1.0, "vol_mult": 0.9}],
        },
        {
            "key": "base",
            "name": "기본",
            "description": "시장 기대수익률(CAPM)과 최근 추세를 반영한 평균적 흐름",
            "prior": 0.40,
            "tilt": 0.0,
            "phases": [{"frac": 1.0, "drift_spread": 0.0, "vol_mult": 1.0}],
        },
        {
            "key": "bear",
            "name": "약세 전환",
            "description": "실적 둔화·밸류에이션 부담으로 하락 흐름이 이어지는 경우",
            "prior": 0.20,
            "tilt": -1.0,
            "phases": [{"frac": 1.0, "drift_spread": -1.5, "vol_mult": 1.25}],
        },
        {
            "key": "dip_recovery",
            "name": "조정 후 반등",
            "description": "초반 1/3 구간 급한 조정 이후 회복하는 V자 흐름",
            "prior": 0.10,
            "tilt": 0.0,
            "phases": [
                {"frac": 0.35, "drift_spread": -2.5, "vol_mult": 1.4},
                {"frac": 0.65, "drift_spread": 1.2, "vol_mult": 1.0},
            ],
        },
        {
            "key": "crash",
            "name": "급락 충격",
            "description": "시장 위기·악재로 급락(점프)이 반복되는 스트레스 상황",
            "prior": 0.05,
            "tilt": -1.0,
            "phases": [
                {
                    "frac": 1.0,
                    "drift_spread": -1.0,
                    "vol_mult": 1.6,
                    "jump_rate": 3.0,
                    "jump_mean": -0.08,
                    "jump_std": 0.05,
                }
            ],
        },
    ],
}


@dataclass
class Calibration:
    """종목 통계로부터 도출한 시뮬레이션 기준값."""

    risk_free: float
    erp: float
    beta: float
    capm_drift: float       # rf + β × ERP
    trend_drift: float      # 최근 1년 추세가 그대로 이어질 때의 연 기대수익률
    trend_weight: float
    base_drift: float       # 기본 시나리오 기대수익률
    spread: float           # 시나리오 간 기대수익률 간격
    vol_now: float          # 현재(EWMA) 변동성
    vol_long: float         # 장기 평균 변동성 (회귀 목표)
    vol_halflife: float     # 변동성 회귀 반감기 (거래일)
    tail_df: float          # 수익률 꼬리 두께 (Student-t 자유도)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase:
    frac: float
    drift: float            # 연 기대수익률 (확산 부분, 산술)
    vol_mult: float = 1.0
    jump_rate: float = 0.0  # 연간 기대 점프 횟수
    jump_mean: float = 0.0  # 점프 크기 평균 (로그수익률)
    jump_std: float = 0.0   # 점프 크기 표준편차


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    prior: float
    tilt: float
    phases: list[Phase]
    probability: float = 0.0
    color_slot: int = 0

    @property
    def avg_drift(self) -> float:
        """국면 비중 가중 평균 기대수익률 (점프 제외)."""
        return sum(ph.frac * ph.drift for ph in self.phases)

    @property
    def jump_drag(self) -> float:
        """점프로 인한 연간 기대 수익률 영향 (근사)."""
        total = 0.0
        for ph in self.phases:
            if ph.jump_rate > 0:
                total += ph.frac * ph.jump_rate * (math.exp(ph.jump_mean + 0.5 * ph.jump_std**2) - 1.0)
        return total

    def to_dict(self) -> dict:
        out = asdict(self)
        out["avg_drift"] = self.avg_drift
        out["jump_drag"] = self.jump_drag
        return out


def calibrate(
    stats: MarketStats,
    risk_free: float,
    erp: float = DEFAULT_ERP,
    trend_weight: float = DEFAULT_TREND_WEIGHT,
    vol_halflife: float = DEFAULT_VOL_HALFLIFE,
) -> Calibration:
    """기본 기대수익률 = (1-w)·CAPM + w·추세, 시나리오 폭 = 장기 변동성의 절반."""
    beta = stats.beta if stats.beta is not None and math.isfinite(stats.beta) else 1.0
    beta_c = min(max(beta, 0.3), 3.0)
    capm = risk_free + beta_c * erp

    vol_1y = stats.vol_1y if math.isfinite(stats.vol_1y) else stats.vol_long
    vol_long = math.sqrt(0.5 * (stats.vol_long**2 + vol_1y**2))
    slope = stats.slope_1y if math.isfinite(stats.slope_1y) else 0.0
    trend = min(max(slope + 0.5 * vol_1y**2, -0.5), 0.8)

    w = min(max(trend_weight, 0.0), 1.0)
    base = (1.0 - w) * capm + w * trend
    spread = min(max(0.5 * vol_long, 0.06), 0.30)
    vol_now = stats.vol_now if math.isfinite(stats.vol_now) else vol_long

    return Calibration(
        risk_free=risk_free,
        erp=erp,
        beta=beta,
        capm_drift=capm,
        trend_drift=trend,
        trend_weight=w,
        base_drift=base,
        spread=spread,
        vol_now=vol_now,
        vol_long=vol_long,
        vol_halflife=vol_halflife,
        tail_df=stats.tail_df,
    )


def load_spec(path: str | Path | None) -> dict:
    """시나리오 설정 JSON을 읽습니다. 경로가 없으면 기본 설정."""
    if path is None:
        return copy.deepcopy(DEFAULT_SPEC)
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    if "scenarios" not in spec or not spec["scenarios"]:
        raise ValueError(f"{path}: 'scenarios' 목록이 비어 있습니다")
    return spec


def build_scenarios(spec: dict, calib: Calibration) -> list[Scenario]:
    """설정(spec)과 보정값으로 실제 수치가 채워진 시나리오 목록을 만듭니다."""
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for idx, raw in enumerate(spec["scenarios"]):
        key = str(raw.get("key") or f"s{idx + 1}")
        if key in seen:
            raise ValueError(f"시나리오 key '{key}' 가 중복되었습니다")
        seen.add(key)
        phases_raw = raw.get("phases")
        if not phases_raw:
            # 단일 국면 축약형: 시나리오 자체에 drift 관련 필드를 둔 경우
            phases_raw = [{k: raw[k] for k in raw if k in _PHASE_KEYS}]
        phases = [_build_phase(ph, calib, key) for ph in phases_raw]
        total = sum(ph.frac for ph in phases)
        if total <= 0:
            raise ValueError(f"시나리오 '{key}': 국면 비중(frac)의 합이 0 입니다")
        for ph in phases:
            ph.frac /= total
        name = str(raw.get("name", key))
        if name in {sc.name for sc in scenarios}:
            name = f"{name} ({key})"
        prior = float(raw.get("prior", raw.get("probability", 1.0)))
        if prior < 0:
            raise ValueError(f"시나리오 '{key}': prior 는 0 이상이어야 합니다")
        scenarios.append(
            Scenario(
                key=key,
                name=name,
                description=str(raw.get("description", "")),
                prior=prior,
                tilt=float(raw.get("tilt", 0.0)),
                phases=phases,
                color_slot=idx,
            )
        )
    return scenarios


_PHASE_KEYS = {"frac", "drift", "drift_spread", "vol_mult", "jump_rate", "jump_mean", "jump_std"}


def _build_phase(raw: dict, calib: Calibration, key: str) -> Phase:
    unknown = set(raw) - _PHASE_KEYS
    if unknown:
        raise ValueError(f"시나리오 '{key}': 알 수 없는 국면 필드 {sorted(unknown)}")
    if "drift" in raw:
        drift = float(raw["drift"])
    else:
        drift = calib.base_drift + float(raw.get("drift_spread", 0.0)) * calib.spread
    phase = Phase(
        frac=float(raw.get("frac", 1.0)),
        drift=drift,
        vol_mult=float(raw.get("vol_mult", 1.0)),
        jump_rate=float(raw.get("jump_rate", 0.0)),
        jump_mean=float(raw.get("jump_mean", 0.0)),
        jump_std=float(raw.get("jump_std", 0.0)),
    )
    if phase.frac < 0 or phase.vol_mult <= 0 or phase.jump_rate < 0 or phase.jump_std < 0:
        raise ValueError(f"시나리오 '{key}': frac/vol_mult/jump 값이 올바르지 않습니다")
    return phase


def assign_probabilities(
    scenarios: list[Scenario],
    trend_score: float,
    tilt_strength: float = DEFAULT_TILT_STRENGTH,
    fixed: bool = False,
) -> None:
    """p_i ∝ prior_i × exp(tilt_i × 추세점수 × 강도). fixed=True 면 prior 를 정규화만 합니다."""
    weights = []
    for sc in scenarios:
        w = sc.prior if fixed else sc.prior * math.exp(sc.tilt * trend_score * tilt_strength)
        weights.append(w)
    total = sum(weights)
    if total <= 0:
        raise ValueError("시나리오 확률(prior)의 합이 0 입니다")
    for sc, w in zip(scenarios, weights):
        sc.probability = w / total


def spec_to_json(spec: dict | None = None) -> str:
    return json.dumps(spec or DEFAULT_SPEC, ensure_ascii=False, indent=2)


def scenario_spec_with_fields(spec: dict) -> dict:
    """기본값을 채운 설정 사본 (tilt_strength 등)."""
    out = copy.deepcopy(spec)
    out.setdefault("tilt_strength", DEFAULT_TILT_STRENGTH)
    out.setdefault("fixed_probabilities", False)
    return out


__all__ = [
    "Calibration",
    "Phase",
    "Scenario",
    "DEFAULT_SPEC",
    "calibrate",
    "load_spec",
    "build_scenarios",
    "assign_probabilities",
    "spec_to_json",
    "scenario_spec_with_fields",
]
