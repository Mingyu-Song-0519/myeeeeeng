# 🧠 AI 투자 비서 구현 기획안 (Clean Architecture)

> **Status**: 📋 계획 검토 대기
> **Created**: 2025-12-25
> **Reference**: [final_ai_development_plan.md](file:///D:/Stock/work_plan/final_ai_development_plan.md)

---

## 📋 프로젝트 개요

### 목표
종목을 스스로 분석하고, 매수/매도 타이밍을 알려주는 **자율형 AI 투자 비서** 구현

### 핵심 원칙
1. **Zero Cost**: Google Gemini 무료 API 사용 (분당 60회, 일 1,500회)
2. **Clean Architecture**: 의존성 역전 원칙(DIP) 준수
3. **확장성**: 추후 Local LLM 하이브리드 아키텍처 지원

---

## 🏗️ Clean Architecture 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    🖥️ Presentation Layer                    │
│  src/dashboard/views/ai_analysis_view.py                   │
│  src/dashboard/views/screener_view.py                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    📦 Application Layer                     │
│  src/services/investment_report_service.py                 │
│  src/services/signal_generator_service.py                  │
│  src/services/screener_service.py                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    🧠 Domain Layer                          │
│  src/domain/ai_report/entities/investment_report.py        │
│  src/domain/ai_report/repositories/interfaces.py           │
│  src/domain/signal/entities/trading_signal.py              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    💾 Infrastructure Layer                  │
│  src/infrastructure/external/gemini_client.py              │
│  src/infrastructure/external/pykrx_gateway.py              │
│  src/infrastructure/repositories/signal_repository.py      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Phase A: The Brain (AI Integration)

### 목표
Google Gemini API를 연동하여 종목별 투자 분석 리포트 생성

### 구현 파일 목록

---

#### [NEW] Infrastructure: `src/infrastructure/external/gemini_client.py`

LLM API 호출을 추상화하는 클라이언트

```python
from abc import ABC, abstractmethod
from typing import Optional

class ILLMClient(ABC):
    """LLM 클라이언트 인터페이스 (DIP)"""
    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = None) -> str:
        pass

class GeminiClient(ILLMClient):
    """Google Gemini API 클라이언트"""
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate(self, prompt: str, system_instruction: str = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text
```

---

#### [NEW] Domain: `src/domain/ai_report/entities/investment_report.py`

AI 분석 리포트 엔티티

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class SignalType(Enum):
    STRONG_BUY = "강력 매수"
    BUY = "매수"
    HOLD = "보유"
    SELL = "매도"
    STRONG_SELL = "강력 매도"

@dataclass
class InvestmentReport:
    ticker: str
    stock_name: str
    signal: SignalType
    confidence_score: float  # 0-100
    summary: str  # AI 분석 요약
    reasoning: str  # 상세 논리
    generated_at: datetime
    
    @property
    def is_actionable(self) -> bool:
        """실행 가능한 신호인지 (신뢰도 80% 이상)"""
        return self.confidence_score >= 80
```

---

#### [NEW] Application: `src/services/investment_report_service.py`

AI 리포트 생성 유즈케이스

```python
class InvestmentReportService:
    def __init__(
        self,
        llm_client: ILLMClient,
        stock_repo: IStockRepository,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None
    ):
        self.llm = llm_client
        self.stock_repo = stock_repo
        self.sentiment = sentiment_analyzer
    
    def generate_report(self, ticker: str) -> InvestmentReport:
        """종목 분석 리포트 생성"""
        # 1. 데이터 수집
        stock_data = self.stock_repo.get_stock_data(ticker, "1mo")
        technical = self._get_technical_summary(stock_data)
        sentiment_score = self.sentiment.analyze(ticker) if self.sentiment else None
        
        # 2. 프롬프트 구성
        prompt = self._build_analyst_prompt(ticker, technical, sentiment_score)
        
        # 3. AI 생성
        response = self.llm.generate(prompt)
        
        # 4. 파싱 및 반환
        return self._parse_response(ticker, response)
```

---

#### [NEW] Presentation: `src/dashboard/views/ai_analysis_view.py`

AI 분석 UI 컴포넌트

```python
def render_ai_analysis_button(ticker: str, stock_name: str):
    """AI 분석 버튼 및 결과 표시"""
    if st.button("🤖 AI 분석 요청", key=f"ai_{ticker}"):
        with st.spinner("AI가 분석 중입니다..."):
            service = _get_report_service()
            report = service.generate_report(ticker)
            
            # 결과 표시
            _display_report(report)

def _display_report(report: InvestmentReport):
    """리포트 카드 UI"""
    signal_colors = {
        SignalType.STRONG_BUY: "green",
        SignalType.BUY: "lightgreen",
        SignalType.HOLD: "gray",
        SignalType.SELL: "orange",
        SignalType.STRONG_SELL: "red"
    }
    # ... UI 렌더링
```

---

## 🚀 Phase B: The Context (Data & Signal Logic)

### 목표
외국인/기관 수급 데이터 + 펀더멘털 + 신호 생성 로직 구현

### 구현 파일 목록

---

#### [NEW] Infrastructure: `src/infrastructure/external/pykrx_gateway.py`

한국 주식 수급 데이터 수집

```python
class PyKRXGateway:
    """pykrx를 이용한 한국 주식 데이터 수집"""
    
    def get_investor_trading(self, ticker: str, days: int = 20) -> pd.DataFrame:
        """투자자별 매매동향 (외국인/기관/개인)"""
        from pykrx import stock
        # ticker에서 .KS, .KQ 제거
        code = ticker.replace(".KS", "").replace(".KQ", "")
        return stock.get_market_trading_value_by_date(
            start_date, end_date, code
        )
```

---

#### [NEW] Domain: `src/domain/signal/entities/trading_signal.py`

매매 신호 엔티티

```python
@dataclass
class TradingSignal:
    ticker: str
    signal_type: SignalType
    confidence: float
    triggers: List[str]  # 발동 조건들
    generated_at: datetime
    
    # 신호 발동 조건
    ai_prediction_confident: bool  # AI 예측 신뢰도 80%+
    sentiment_positive: bool       # 감성 점수 0.7+
    volume_spike_detected: bool    # 거래량 급등
    institution_buying: bool       # 기관 순매수
```

---

#### [NEW] Application: `src/services/signal_generator_service.py`

매매 신호 생성 서비스

```python
class SignalGeneratorService:
    """매매 신호 생성기 (라씨매매신호 스타일)"""
    
    def generate_signal(self, ticker: str) -> TradingSignal:
        """종합 매매 신호 생성"""
        # 1. AI 예측 신뢰도 체크
        ai_confident = self._check_ai_confidence(ticker)
        
        # 2. 감성 점수 체크
        sentiment_positive = self._check_sentiment(ticker)
        
        # 3. 거래량 급등 체크
        volume_spike = self._check_volume_spike(ticker)
        
        # 4. 기관 수급 체크
        inst_buying = self._check_institution_buying(ticker)
        
        # 5. 종합 판단
        triggers = []
        if ai_confident: triggers.append("AI 신뢰도 80%+")
        if sentiment_positive: triggers.append("감성 긍정적")
        if volume_spike: triggers.append("거래량 급등")
        if inst_buying: triggers.append("기관 매수세")
        
        # 3개 이상 충족 시 강력 매수
        if len(triggers) >= 3:
            return TradingSignal(signal_type=SignalType.STRONG_BUY, ...)
```

---

## 🚀 Phase C: The Hands (Screener & Personalization)

### 목표
AI 기반 종목 발굴 + 사용자 성향 맞춤 추천

### 구현 파일 목록

---

#### [NEW] Application: `src/services/screener_service.py`

종목 스크리너 서비스

```python
class ScreenerService:
    """AI 종목 스크리너 (매일 아침 추천주)"""
    
    def run_daily_screen(self, user_id: str) -> List[StockRecommendation]:
        """일일 스크리닝 실행"""
        # 1. 전체 종목 풀 가져오기
        all_tickers = self._get_stock_universe()
        
        # 2. 기본 필터링 (RSI, PBR 등)
        filtered = self._apply_base_filters(all_tickers)
        
        # 3. AI 점수 계산
        scored = self._calculate_ai_scores(filtered)
        
        # 4. 사용자 프로필 기반 재정렬
        profile = self.profile_repo.load(user_id)
        personalized = self._personalize_ranking(scored, profile)
        
        return personalized[:5]  # Top 5 추천
```

---

#### [NEW] Presentation: `src/dashboard/views/screener_view.py`

AI 스크리너 UI

```python
def render_morning_picks():
    """오늘의 AI 추천주"""
    st.header("🌅 AI 모닝 픽")
    
    service = _get_screener_service()
    picks = service.run_daily_screen(user_id)
    
    for pick in picks:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"{pick.stock_name}")
            with col2:
                st.metric("AI 점수", f"{pick.ai_score:.0f}")
            with col3:
                st.button("상세 분석", key=f"detail_{pick.ticker}")
```

---

## 📁 신규 파일 구조 요약

```
src/
├── domain/
│   ├── ai_report/
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   └── investment_report.py      # [NEW] AI 리포트 엔티티
│   │   └── repositories/
│   │       └── interfaces.py             # [NEW] 리포트 저장소 인터페이스
│   └── signal/
│       ├── __init__.py
│       └── entities/
│           └── trading_signal.py         # [NEW] 매매 신호 엔티티
│
├── services/
│   ├── investment_report_service.py      # [NEW] AI 리포트 서비스
│   ├── signal_generator_service.py       # [NEW] 신호 생성 서비스
│   └── screener_service.py               # [NEW] 스크리너 서비스
│
├── infrastructure/
│   └── external/
│       ├── gemini_client.py              # [NEW] Gemini API 클라이언트
│       └── pykrx_gateway.py              # [NEW] pykrx 데이터 게이트웨이
│
└── dashboard/
    └── views/
        ├── ai_analysis_view.py           # [NEW] AI 분석 UI
        └── screener_view.py              # [NEW] 스크리너 UI
```

---

## ✅ 검증 계획

### 단위 테스트
```bash
# Gemini 클라이언트 테스트
python -c "from src.infrastructure.external.gemini_client import GeminiClient; print('OK')"

# 서비스 레이어 테스트
python -c "from src.services.investment_report_service import InvestmentReportService; print('OK')"
```

### 통합 테스트
1. AI 분석 버튼 클릭 → 리포트 생성 확인
2. 스크리너 실행 → Top 5 종목 표시 확인
3. 매매 신호 → 조건 충족 시 알림 확인

---

## 📅 예상 일정

| Phase | Task | Duration | Priority |
|-------|------|----------|----------|
| A-1 | Gemini 클라이언트 구현 | 1일 | 🔥 높음 |
| A-2 | 리포트 서비스 & UI | 2일 | 🔥 높음 |
| B-1 | 수급 데이터 연동 (pykrx) | 2일 | ⚡ 중간 |
| B-2 | 신호 생성 로직 | 2일 | 🔥 높음 |
| C-1 | 스크리너 서비스 | 2일 | ⚡ 중간 |
| C-2 | 개인화 엔진 | 1일 | 💡 낮음 |

**총 예상 소요**: 10-12일

---

## 🎯 다음 단계

**Phase A-1 (Gemini 클라이언트)** 구현부터 시작합니다.

1. `google-generativeai` 라이브러리 설치
2. `GeminiClient` 어댑터 구현
3. API 키 설정 (Streamlit Secrets)
4. 간단한 테스트 프롬프트 실행

---
---

# 📋 AI 투자 비서 기획안 검토 및 개선 권장사항

> **검토일**: 2025-12-25
> **검토 기준**: Feature Planner Skill + Clean Architecture + 기존 인프라 통합
> **검토자**: Claude Code (Sonnet 4.5)

---

## ✅ 강점 분석

### 1. Clean Architecture 완벽 준수 ⭐⭐⭐⭐⭐

**평가**:
- ✅ Domain/Infrastructure/Application/Presentation 4계층 명확히 분리
- ✅ 의존성 역전 원칙(DIP) 철저히 적용 (ILLMClient 인터페이스)
- ✅ Rich Domain Model (InvestmentReport, TradingSignal)
- ✅ Infrastructure 추상화 (Gemini → 추후 Local LLM 교체 가능)

**코드 증거**:
```python
# ✅ 우수 사례: 인터페이스 기반 설계
class ILLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = None) -> str:
        pass

class GeminiClient(ILLMClient):  # ← DIP 준수
    # 구현...
```

**기대 효과**:
- Local LLM (Ollama/LLaMA) 전환 시 서비스 레이어 수정 불필요
- 테스트 시 Mock LLM 주입 가능 → TDD 가능

---

### 2. 단계적 구현 계획 ⭐⭐⭐⭐

**평가**:
- ✅ Phase A (AI 통합) → Phase B (데이터/신호) → Phase C (스크리너) 순차 진행
- ✅ 각 Phase별 명확한 산출물
- ✅ 10-12일 일정 현실적

**강점**:
- MVP (Phase A-2)를 2-3일 내 완료 가능
- 사용자 피드백 조기 수집 가능

---

### 3. 기존 시스템과의 자연스러운 연결점 ⭐⭐⭐⭐

**평가**:
- ✅ SentimentAnalysisService 기존 존재 확인
- ✅ RecommendationService와 통합 가능
- ✅ InvestorProfile 엔티티 재사용 가능

---

## 🔴 중대한 누락 사항

### 1. Phase 20 투자 성향 프로필 연동 미정의 (우선순위: ⭐⭐⭐⭐⭐)

**문제**:
- ✅ AI 리포트 생성 기능은 정의됨
- ❌ **Phase 20 InvestorProfile과의 통합 방안 없음**
- ❌ 사용자 성향에 맞는 AI 추천 개인화 전략 부재
- ❌ 투자 성향에 따른 리포트 톤 조절 로직 없음

**영향**:
- Phase 20에서 구축한 투자 성향 프로필이 활용되지 않음
- 모든 사용자에게 동일한 AI 분석 제공 → 차별화 요소 부족
- 안정형 투자자에게 고위험 종목 추천 가능 → 사용자 불만

**해결 방안**:

#### Option A: InvestmentReportService에 프로필 기반 개인화 (권장)

```python
# src/services/investment_report_service.py (수정)
class InvestmentReportService:
    def __init__(
        self,
        llm_client: ILLMClient,
        stock_repo: IStockRepository,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        profile_repo: Optional[IProfileRepository] = None  # ← Phase 20 통합
    ):
        self.llm = llm_client
        self.stock_repo = stock_repo
        self.sentiment = sentiment_analyzer
        self.profile_repo = profile_repo  # ← NEW

    def generate_personalized_report(
        self,
        ticker: str,
        user_id: str  # ← NEW
    ) -> InvestmentReport:
        """사용자 성향 기반 맞춤 AI 리포트 생성"""
        # 1. 기본 데이터 수집
        stock_data = self.stock_repo.get_stock_data(ticker, "1mo")
        technical = self._get_technical_summary(stock_data)

        # 2. 사용자 프로필 로드
        profile = self.profile_repo.load(user_id) if self.profile_repo else None

        # 3. 프로필 기반 프롬프트 조정
        prompt = self._build_personalized_prompt(
            ticker,
            technical,
            profile  # ← 성향에 따라 프롬프트 톤 조절
        )

        # 4. AI 생성
        response = self.llm.generate(prompt)

        # 5. 성향 적합도 검증
        report = self._parse_response(ticker, response)

        if profile:
            report = self._adjust_for_profile(report, profile)  # ← 후처리

        return report

    def _build_personalized_prompt(
        self,
        ticker: str,
        technical: dict,
        profile: Optional[InvestorProfile]
    ) -> str:
        """프로필 기반 프롬프트 구성"""
        base_prompt = f"""
종목: {ticker}
기술적 분석: {technical}

위 데이터를 기반으로 투자 분석 리포트를 작성해주세요.
"""

        # 프로필에 따른 지시 추가
        if profile:
            risk_value = profile.risk_tolerance.value

            if risk_value <= 40:  # 안정형/안정추구형
                base_prompt += """
[중요] 이 사용자는 안정적인 투자를 선호합니다.
- 변동성이 큰 종목은 신중하게 평가하세요.
- 리스크 요인을 명확히 강조하세요.
- 배당 수익률, PBR 등 안정성 지표를 중심으로 분석하세요.
"""
            elif risk_value > 60:  # 적극투자형/공격투자형
                base_prompt += """
[중요] 이 사용자는 공격적인 투자를 선호합니다.
- 성장 가능성과 모멘텀을 중심으로 분석하세요.
- 높은 수익률 기회를 강조하세요.
- 단기 트레이딩 관점도 포함하세요.
"""

            # 선호 섹터 반영
            if profile.preferred_sectors:
                sectors_str = ", ".join(profile.preferred_sectors)
                base_prompt += f"""
[참고] 사용자 선호 섹터: {sectors_str}
→ 해당 섹터와의 연관성을 분석에 포함하세요.
"""

        return base_prompt

    def _adjust_for_profile(
        self,
        report: InvestmentReport,
        profile: InvestorProfile
    ) -> InvestmentReport:
        """프로필에 맞지 않는 추천 조정"""
        # 예: 안정형 투자자에게 STRONG_BUY가 나왔지만 고변동성 종목인 경우
        stock_info = self.stock_repo.get_stock_info(report.ticker)
        volatility = stock_info.get('volatility', 0.3)

        risk_value = profile.risk_tolerance.value

        # 안정형 + 고변동성 → 신호 하향 조정
        if risk_value <= 40 and volatility > 0.35:
            if report.signal == SignalType.STRONG_BUY:
                report.signal = SignalType.BUY
                report.reasoning += "\n\n⚠️ 주의: 이 종목은 변동성이 높아 안정형 투자자에게는 신중한 접근이 필요합니다."
                report.confidence_score *= 0.8  # 신뢰도 하향

        # 공격형 + 저변동성 → 경고 추가
        if risk_value > 60 and volatility < 0.2:
            report.reasoning += "\n\n💡 참고: 이 종목은 안정적이지만 단기 수익률은 제한적일 수 있습니다."

        return report
```

**추가 필요 작업**:
- Phase A-2: `InvestmentReportService`에 `profile_repo` 의존성 추가
- Phase A-2: `_build_personalized_prompt()` 메서드 구현
- Phase A-2: `_adjust_for_profile()` 후처리 로직 구현
- Phase A-2: UI에 "내 성향 맞춤 분석" 토글 추가

---

### 2. Phase 21 Market Buzz 연동 미정의 (우선순위: ⭐⭐⭐⭐⭐)

**문제**:
- ✅ AI 리포트 생성 기능은 정의됨
- ❌ **Phase 21 Market Buzz 데이터가 AI 프롬프트에 포함되지 않음**
- ❌ 거래량 급증, Buzz 점수 정보가 AI 분석에 반영되지 않음
- ❌ Screener가 Buzz 점수를 고려하지 않음

**영향**:
- Phase 21에서 구축한 Market Buzz 시스템이 활용되지 않음
- AI가 시장 관심도를 모르는 채로 분석 → 불완전한 리포트
- 거래량 급증 중인 종목을 AI가 인지하지 못함

**해결 방안**:

#### Option A: AI 프롬프트에 Market Buzz 데이터 포함 (권장)

```python
# src/services/investment_report_service.py (추가)
class InvestmentReportService:
    def __init__(
        self,
        llm_client: ILLMClient,
        stock_repo: IStockRepository,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        profile_repo: Optional[IProfileRepository] = None,
        market_buzz_service: Optional[MarketBuzzService] = None  # ← Phase 21 통합
    ):
        self.llm = llm_client
        self.stock_repo = stock_repo
        self.sentiment = sentiment_analyzer
        self.profile_repo = profile_repo
        self.market_buzz_service = market_buzz_service  # ← NEW

    def generate_report(self, ticker: str, user_id: str = None) -> InvestmentReport:
        """종목 분석 리포트 생성 (Market Buzz 반영)"""
        # 1. 데이터 수집
        stock_data = self.stock_repo.get_stock_data(ticker, "1mo")
        technical = self._get_technical_summary(stock_data)
        sentiment_score = self.sentiment.analyze(ticker) if self.sentiment else None

        # 2. Market Buzz 데이터 수집 (Phase 21)
        buzz_data = None
        if self.market_buzz_service:
            try:
                buzz_score_obj = self.market_buzz_service.calculate_buzz_score(ticker)
                if buzz_score_obj:
                    buzz_data = {
                        'base_score': buzz_score_obj.base_score,
                        'heat_level': buzz_score_obj.heat_level,
                        'volume_ratio': buzz_score_obj.volume_ratio,
                        'volatility_ratio': buzz_score_obj.volatility_ratio
                    }
            except Exception as e:
                logger.warning(f"Failed to get buzz data for {ticker}: {e}")

        # 3. 프롬프트 구성 (Buzz 데이터 포함)
        prompt = self._build_analyst_prompt(
            ticker,
            technical,
            sentiment_score,
            buzz_data  # ← NEW
        )

        # 4. AI 생성
        response = self.llm.generate(prompt)

        # 5. 파싱 및 반환
        return self._parse_response(ticker, response)

    def _build_analyst_prompt(
        self,
        ticker: str,
        technical: dict,
        sentiment_score: Optional[float],
        buzz_data: Optional[dict]
    ) -> str:
        """분석가 스타일 프롬프트 구성"""
        prompt = f"""
당신은 전문 주식 애널리스트입니다. 아래 데이터를 분석하여 투자 의견을 제시하세요.

종목: {ticker}

기술적 분석:
- RSI: {technical.get('rsi', 'N/A')}
- MACD: {technical.get('macd', 'N/A')}
- 볼린저 밴드: {technical.get('bbands', 'N/A')}
"""

        if sentiment_score is not None:
            prompt += f"""
뉴스 감성 분석:
- 감성 점수: {sentiment_score:.2f} (0=매우 부정적, 1=매우 긍정적)
"""

        # Phase 21 Market Buzz 데이터 추가
        if buzz_data:
            prompt += f"""
시장 관심도 (Market Buzz):
- Buzz 점수: {buzz_data['base_score']:.0f}/100
- 시장 열기: {buzz_data['heat_level']} {"🔥" if buzz_data['heat_level'] == "HOT" else ""}
- 거래량 비율: {buzz_data['volume_ratio']:.2f}x (평균 대비)
- 변동성 비율: {buzz_data['volatility_ratio']:.2f}x (평균 대비)

{"⚠️ 주의: 최근 거래량이 급증했습니다. 단기 모멘텀이 강합니다." if buzz_data['volume_ratio'] > 2.0 else ""}
"""

        prompt += """
[분석 요청]
1. 종합 평가 (매수/보유/매도)
2. 신뢰도 (0-100점)
3. 핵심 근거 (3-5줄 요약)
4. 상세 논리 (기술적/감성적/시장 관심도 종합)

출력 형식:
```
신호: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
신뢰도: [0-100]
요약: [3-5줄 요약]
논리: [상세 분석]
```
"""
        return prompt
```

**추가 필요 작업**:
- Phase A-2: `InvestmentReportService`에 `market_buzz_service` 의존성 주입
- Phase A-2: `_build_analyst_prompt()`에 Buzz 데이터 포함
- Phase C-1: `ScreenerService`에서 Buzz 점수 높은 종목 우선 선택

---

### 3. 기존 SentimentAnalysisService 활용 미명시 (우선순위: ⭐⭐⭐⭐)

**문제**:
- ✅ 기존 `SentimentAnalysisService` 존재 확인 (sentiment_analysis_service.py)
- ❌ **AI 기획안에서 이를 재구현하려는 듯한 뉘앙스**
- ❌ 기존 서비스 재사용 전략 명시되지 않음

**영향**:
- 중복 개발 위험 → 개발 시간 낭비
- 기존 NewsCollector, SentimentAnalyzer 인프라 미활용

**해결 방안**:

#### 기존 SentimentAnalysisService 직접 주입

```python
# src/services/investment_report_service.py (수정)
from src.services.sentiment_analysis_service import SentimentAnalysisService  # ← 기존 재사용

class InvestmentReportService:
    def __init__(
        self,
        llm_client: ILLMClient,
        stock_repo: IStockRepository,
        sentiment_service: Optional[SentimentAnalysisService] = None,  # ← 기존 서비스 주입
        profile_repo: Optional[IProfileRepository] = None,
        market_buzz_service: Optional[MarketBuzzService] = None
    ):
        self.llm = llm_client
        self.stock_repo = stock_repo
        self.sentiment_service = sentiment_service or SentimentAnalysisService()  # ← 기존 활용
        self.profile_repo = profile_repo
        self.market_buzz_service = market_buzz_service

    def generate_report(self, ticker: str, user_id: str = None) -> InvestmentReport:
        """종목 분석 리포트 생성"""
        # 기존 SentimentAnalysisService 사용
        sentiment_features = self.sentiment_service.get_sentiment_features(
            ticker=ticker,
            lookback_days=7
        )
        sentiment_score = sentiment_features.get('sentiment_score', 0.5)

        # ... AI 프롬프트 구성 시 sentiment_score 활용
```

**추가 필요 작업**:
- Phase A-2: `SentimentAnalysisService` import 및 주입
- Phase A-2: 기획안 문서 수정 (중복 구현 제거)

---

### 4. 한국 vs 미국 주식 구분 처리 미정의 (우선순위: ⭐⭐⭐⭐)

**문제**:
- ❌ **PyKRXGateway는 한국 주식 전용**
- ❌ 미국 주식 데이터 수집 전략 없음
- ❌ AI 프롬프트가 시장별 특성을 고려하지 않음

**영향**:
- 미국 주식에 대해 외국인/기관 수급 데이터 조회 불가 → AI 프롬프트 불완전
- 한국 주식과 미국 주식 혼재 시 오류 발생
- 시장별 다른 분석 프레임워크 필요 (KR: 외국인 수급, US: Insider Trading)

**해결 방안**:

#### Option A: Market Detection + 조건부 데이터 수집

```python
# src/services/investment_report_service.py (추가)
class InvestmentReportService:
    def __init__(
        self,
        llm_client: ILLMClient,
        stock_repo: IStockRepository,
        pykrx_gateway: Optional[PyKRXGateway] = None,  # ← 한국 주식 전용
        us_data_gateway: Optional[USDataGateway] = None,  # ← 미국 주식용 (NEW)
        # ...
    ):
        self.llm = llm_client
        self.stock_repo = stock_repo
        self.pykrx_gateway = pykrx_gateway
        self.us_data_gateway = us_data_gateway

    def generate_report(self, ticker: str, user_id: str = None) -> InvestmentReport:
        """종목 분석 리포트 생성 (시장 자동 감지)"""
        # 1. 시장 감지
        market = self._detect_market(ticker)

        # 2. 시장별 데이터 수집
        if market == "KR":
            # 한국 주식: pykrx 외국인/기관 수급
            investor_data = self.pykrx_gateway.get_investor_trading(ticker) if self.pykrx_gateway else None
            prompt = self._build_kr_prompt(ticker, investor_data)
        else:
            # 미국 주식: yfinance 기본 정보
            # (외국인 수급 대신 Insider Trading, Institutional Ownership 등)
            us_data = self.us_data_gateway.get_institutional_ownership(ticker) if self.us_data_gateway else None
            prompt = self._build_us_prompt(ticker, us_data)

        # 3. AI 생성
        response = self.llm.generate(prompt)
        return self._parse_response(ticker, response)

    def _detect_market(self, ticker: str) -> str:
        """티커에서 시장 자동 판별"""
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            return 'KR'
        elif '.' not in ticker or ticker.endswith('.US'):
            return 'US'
        else:
            # yfinance로 조회하여 확인
            try:
                stock_info = self.stock_repo.get_stock_info(ticker)
                exchange = stock_info.get('exchange', '')
                if 'KRX' in exchange or 'KSE' in exchange:
                    return 'KR'
                else:
                    return 'US'
            except:
                return 'US'  # 기본값

    def _build_kr_prompt(self, ticker: str, investor_data: Optional[pd.DataFrame]) -> str:
        """한국 주식 분석 프롬프트"""
        prompt = f"종목: {ticker} (한국 거래소)\n\n"

        if investor_data is not None and not investor_data.empty:
            foreign_net = investor_data['외국인순매수'].sum()
            inst_net = investor_data['기관순매수'].sum()

            prompt += f"""
투자자별 매매동향 (최근 20일):
- 외국인 순매수: {foreign_net:,.0f}원
- 기관 순매수: {inst_net:,.0f}원

{"✅ 외국인/기관 동반 매수세" if foreign_net > 0 and inst_net > 0 else ""}
{"⚠️ 외국인/기관 동반 매도세" if foreign_net < 0 and inst_net < 0 else ""}
"""

        prompt += "\n위 데이터를 종합하여 투자 의견을 제시하세요."
        return prompt

    def _build_us_prompt(self, ticker: str, us_data: Optional[dict]) -> str:
        """미국 주식 분석 프롬프트"""
        prompt = f"종목: {ticker} (미국 거래소)\n\n"

        if us_data:
            prompt += f"""
기관 보유 현황:
- Institutional Ownership: {us_data.get('institutional_ownership', 0)*100:.1f}%
- Insider Ownership: {us_data.get('insider_ownership', 0)*100:.1f}%
"""

        prompt += "\n위 데이터를 종합하여 투자 의견을 제시하세요."
        return prompt
```

**추가 필요 작업**:
- Phase B-1: `USDataGateway` 구현 (yfinance 기반)
- Phase B-2: 시장 감지 로직 구현
- Phase B-2: 시장별 프롬프트 템플릿 분리

---

### 5. SignalGeneratorService 조건 로직 과도하게 단순 (우선순위: ⭐⭐⭐⭐)

**문제**:
- ✅ 4가지 조건 (AI 신뢰도, 감성, 거래량, 기관 수급) 정의됨
- ❌ **"3개 이상 충족 시 강력 매수"는 너무 기계적**
- ❌ 조건 간 가중치 없음 (AI 신뢰도 80%와 거래량 급증의 중요도가 다름)
- ❌ 시장 상황(상승장/하락장) 고려 없음

**영향**:
- 허위 신호(False Positive) 발생 위험
- 약세장에서도 매수 신호 발생 가능 → 사용자 손실

**해결 방안**:

#### Option A: 가중치 기반 종합 점수 계산

```python
# src/services/signal_generator_service.py (개선)
class SignalGeneratorService:
    """매매 신호 생성기 (라씨매매신호 스타일)"""

    # 조건별 가중치 정의
    WEIGHTS = {
        'ai_confidence': 0.35,     # AI 신뢰도: 35%
        'sentiment': 0.25,         # 감성: 25%
        'volume_spike': 0.20,      # 거래량: 20%
        'institution_buying': 0.20 # 기관 수급: 20%
    }

    def generate_signal(self, ticker: str) -> TradingSignal:
        """종합 매매 신호 생성 (가중치 기반)"""
        # 1. 조건별 점수 계산 (0-100)
        ai_score = self._calculate_ai_score(ticker)
        sentiment_score = self._calculate_sentiment_score(ticker)
        volume_score = self._calculate_volume_score(ticker)
        inst_score = self._calculate_institution_score(ticker)

        # 2. 가중 평균
        composite_score = (
            ai_score * self.WEIGHTS['ai_confidence'] +
            sentiment_score * self.WEIGHTS['sentiment'] +
            volume_score * self.WEIGHTS['volume_spike'] +
            inst_score * self.WEIGHTS['institution_buying']
        )

        # 3. 시장 상황 보정 (Phase 21 Market Buzz 활용)
        market_regime = self._get_market_regime()  # "BULL" / "BEAR" / "NEUTRAL"
        if market_regime == "BEAR":
            composite_score *= 0.7  # 약세장에서는 신호 강도 하향

        # 4. 신호 판정
        if composite_score >= 80:
            signal_type = SignalType.STRONG_BUY
        elif composite_score >= 65:
            signal_type = SignalType.BUY
        elif composite_score >= 40:
            signal_type = SignalType.HOLD
        elif composite_score >= 20:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.STRONG_SELL

        # 5. 발동 조건 문자열 생성
        triggers = []
        if ai_score >= 80: triggers.append(f"AI 신뢰도 {ai_score:.0f}점")
        if sentiment_score >= 70: triggers.append(f"감성 긍정적 {sentiment_score:.0f}점")
        if volume_score >= 70: triggers.append("거래량 급등")
        if inst_score >= 70: triggers.append("기관 매수세")

        return TradingSignal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=composite_score,
            triggers=triggers,
            generated_at=datetime.now(),
            market_regime=market_regime  # ← 시장 상황 추가
        )

    def _calculate_ai_score(self, ticker: str) -> float:
        """AI 신뢰도 점수 (0-100)"""
        # InvestmentReportService에서 최근 리포트 조회
        recent_report = self._get_recent_report(ticker)
        if recent_report and recent_report.confidence_score >= 80:
            return recent_report.confidence_score
        return 0

    def _calculate_sentiment_score(self, ticker: str) -> float:
        """감성 점수 (0-100)"""
        sentiment_features = self.sentiment_service.get_sentiment_features(ticker)
        raw_score = sentiment_features.get('sentiment_score', 0.5)  # 0-1
        return raw_score * 100  # 0-100 변환

    def _calculate_volume_score(self, ticker: str) -> float:
        """거래량 점수 (0-100)"""
        # Phase 21 VolumeAnomaly 활용
        anomalies = self.market_buzz_service.detect_volume_anomalies([ticker], threshold=1.5)
        if anomalies:
            volume_ratio = anomalies[0].volume_ratio
            return min((volume_ratio - 1.0) * 25, 100)  # 1.5x = 12.5점, 5x = 100점
        return 0

    def _calculate_institution_score(self, ticker: str) -> float:
        """기관 수급 점수 (0-100)"""
        investor_data = self.pykrx_gateway.get_investor_trading(ticker, days=20)
        if investor_data is not None and not investor_data.empty:
            inst_net = investor_data['기관순매수'].sum()
            foreign_net = investor_data['외국인순매수'].sum()

            # 외국인+기관 동반 매수: 100점
            if inst_net > 0 and foreign_net > 0:
                return 100
            # 둘 중 하나만 매수: 50점
            elif inst_net > 0 or foreign_net > 0:
                return 50
            # 둘 다 매도: 0점
            else:
                return 0
        return 50  # 데이터 없으면 중립

    def _get_market_regime(self) -> str:
        """시장 상황 판별 (BULL/BEAR/NEUTRAL)"""
        # KOSPI/S&P500 최근 추세로 판단
        # 간단 구현: 20일 이동평균 vs 현재가
        kospi_data = self.stock_repo.get_stock_data("^KS11", "1mo")
        if kospi_data is not None and not kospi_data.empty:
            current_price = kospi_data['Close'].iloc[-1]
            ma20 = kospi_data['Close'].rolling(20).mean().iloc[-1]

            if current_price > ma20 * 1.05:
                return "BULL"
            elif current_price < ma20 * 0.95:
                return "BEAR"

        return "NEUTRAL"
```

**추가 필요 작업**:
- Phase B-2: 가중치 기반 신호 생성 로직 구현
- Phase B-2: 시장 상황 판별 로직 추가
- Phase B-2: 조건별 점수 계산 메서드 구현

---

### 6. 기존 RecommendationService와 중복 가능성 (우선순위: ⭐⭐⭐⭐)

**문제**:
- ✅ 기존 `RecommendationService` 존재 (Phase 20)
- ✅ 신규 `ScreenerService` 기획 (Phase C-1)
- ❌ **두 서비스의 역할 구분 불명확**
- ❌ 통합 전략 없음

**영향**:
- 사용자 혼란 ("추천 종목" vs "AI 모닝 픽" 차이가 뭔가?)
- 중복 개발 가능성

**해결 방안**:

#### Option A: ScreenerService를 RecommendationService의 데이터 소스로 활용

```python
# src/services/recommendation_service.py (수정)
class RecommendationService:
    """맞춤 종목 추천 서비스 (Phase 20 기존)"""

    def __init__(
        self,
        profile_repo: IProfileRepository,
        use_ai_model: bool = True,
        screener_service: Optional[ScreenerService] = None  # ← Phase C 통합
    ):
        self.profile_repo = profile_repo
        self.use_ai_model = use_ai_model
        self.screener_service = screener_service  # ← NEW
        # ...

    def generate_recommendations(
        self,
        profile: InvestorProfile,
        top_n: int = 10,
        use_ai_screener: bool = True  # ← NEW
    ) -> List[RankedStock]:
        """추천 종목 생성"""

        if use_ai_screener and self.screener_service:
            # Phase C ScreenerService 활용
            ai_candidates = self.screener_service.run_daily_screen(profile.user_id)

            # AI 후보를 기존 랭킹 시스템과 결합
            combined = self._merge_ai_and_traditional(ai_candidates, profile)
            return combined[:top_n]
        else:
            # 기존 방식 (StockRankingService 기반)
            return self._generate_traditional_recommendations(profile, top_n)

    def _merge_ai_and_traditional(
        self,
        ai_candidates: List[StockRecommendation],
        profile: InvestorProfile
    ) -> List[RankedStock]:
        """AI 후보 + 기존 랭킹 결합"""
        # AI 점수 (0-100)와 기존 Composite Score (0-100) 가중 평균
        # AI 30%, Traditional 70%
        merged = []

        for ai_rec in ai_candidates:
            traditional_score = self._get_traditional_score(ai_rec.ticker, profile)
            final_score = ai_rec.ai_score * 0.3 + traditional_score * 0.7

            merged.append(RankedStock(
                ticker=ai_rec.ticker,
                stock_name=ai_rec.stock_name,
                composite_score=final_score,
                # ...
            ))

        merged.sort(key=lambda x: x.composite_score, reverse=True)
        return merged
```

**역할 구분**:
- **RecommendationService (Phase 20)**: 사용자 성향 기반 맞춤 추천 (메인 추천 엔진)
- **ScreenerService (Phase C)**: AI 기반 종목 발굴 (보조 데이터 소스)

**UI 통합**:
- "추천 종목" 탭: RecommendationService (AI + Traditional 결합)
- "AI 모닝 픽" 탭: ScreenerService 단독 (순수 AI)

**추가 필요 작업**:
- Phase C-1: `ScreenerService`를 `RecommendationService`에 주입
- Phase C-2: `_merge_ai_and_traditional()` 메서드 구현
- Phase C-2: UI에 "AI 강화 모드" 토글 추가

---

### 7. 프롬프트 엔지니어링 전략 부재 (우선순위: ⭐⭐⭐)

**문제**:
- ❌ **프롬프트 예시가 너무 간단함**
- ❌ Few-shot Learning 전략 없음
- ❌ 프롬프트 버전 관리 방안 없음
- ❌ AI 응답 파싱 실패 시 Fallback 전략 없음

**영향**:
- AI가 요청한 형식으로 답변하지 않음 → 파싱 오류 빈발
- 프롬프트 개선 시 버전 추적 불가
- 일관성 없는 AI 응답 품질

**해결 방안**:

#### Option A: 프롬프트 템플릿 관리 시스템

```python
# src/infrastructure/external/prompt_templates.py (NEW)
from typing import Dict, Optional
from enum import Enum

class PromptVersion(Enum):
    V1_BASIC = "v1_basic"
    V2_FEWSHOT = "v2_fewshot"
    V3_COT = "v3_cot"  # Chain-of-Thought

class PromptTemplateManager:
    """프롬프트 템플릿 관리자"""

    TEMPLATES = {
        PromptVersion.V1_BASIC: """
당신은 전문 주식 애널리스트입니다. 아래 데이터를 분석하여 투자 의견을 제시하세요.

{data_section}

[분석 요청]
1. 종합 평가 (매수/보유/매도)
2. 신뢰도 (0-100점)
3. 핵심 근거 (3-5줄 요약)

출력 형식 (반드시 준수):
```
신호: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
신뢰도: [0-100 사이 정수]
요약: [요약 내용]
```
""",

        PromptVersion.V2_FEWSHOT: """
당신은 전문 주식 애널리스트입니다. 아래 예시를 참고하여 분석하세요.

[예시 1]
종목: 삼성전자
RSI: 45, MACD: 매수, 감성: 0.7
→ 신호: BUY, 신뢰도: 75, 요약: 기술적 지표 양호, 감성 긍정적

[예시 2]
종목: 카카오
RSI: 72, MACD: 매도, 감성: 0.3
→ 신호: SELL, 신뢰도: 80, 요약: 과매수 구간, 감성 부정적

[실제 분석 대상]
{data_section}

출력 형식 (반드시 준수):
```
신호: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
신뢰도: [0-100]
요약: [요약]
```
""",

        PromptVersion.V3_COT: """
당신은 전문 주식 애널리스트입니다. 단계별로 사고하세요.

{data_section}

[분석 단계]
1단계: 기술적 지표 평가
2단계: 감성 분석 평가
3단계: 시장 관심도 평가
4단계: 종합 판단

각 단계를 명확히 구분하여 작성한 후, 최종 결론을 내리세요.

출력 형식:
```
1단계: [기술적 분석]
2단계: [감성 분석]
3단계: [시장 관심도]
4단계 (최종):
신호: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
신뢰도: [0-100]
요약: [요약]
```
"""
    }

    @classmethod
    def get_template(
        cls,
        version: PromptVersion = PromptVersion.V2_FEWSHOT
    ) -> str:
        """프롬프트 템플릿 조회"""
        return cls.TEMPLATES[version]

    @classmethod
    def build_prompt(
        cls,
        version: PromptVersion,
        data_section: str
    ) -> str:
        """데이터 섹션을 템플릿에 삽입"""
        template = cls.get_template(version)
        return template.format(data_section=data_section)


# src/services/investment_report_service.py (수정)
from src.infrastructure.external.prompt_templates import PromptTemplateManager, PromptVersion

class InvestmentReportService:
    def __init__(
        self,
        llm_client: ILLMClient,
        # ...
        prompt_version: PromptVersion = PromptVersion.V2_FEWSHOT  # ← 버전 선택 가능
    ):
        self.llm = llm_client
        self.prompt_version = prompt_version

    def generate_report(self, ticker: str, user_id: str = None) -> InvestmentReport:
        """종목 분석 리포트 생성"""
        # 1. 데이터 섹션 구성
        data_section = self._build_data_section(ticker)

        # 2. 프롬프트 생성 (버전 관리)
        prompt = PromptTemplateManager.build_prompt(
            version=self.prompt_version,
            data_section=data_section
        )

        # 3. AI 생성 (재시도 로직)
        try:
            response = self.llm.generate(prompt)
            report = self._parse_response(ticker, response)
        except ParsingError as e:
            logger.warning(f"Parsing failed, retrying with V1_BASIC: {e}")
            # Fallback: 더 간단한 프롬프트로 재시도
            prompt_fallback = PromptTemplateManager.build_prompt(
                version=PromptVersion.V1_BASIC,
                data_section=data_section
            )
            response = self.llm.generate(prompt_fallback)
            report = self._parse_response(ticker, response)

        return report
```

**추가 필요 작업**:
- Phase A-2: `PromptTemplateManager` 구현
- Phase A-2: Few-shot 예시 데이터 준비
- Phase A-2: 파싱 실패 시 Fallback 로직 구현

---

### 8. API 비용 관리 및 Rate Limiting 전략 부재 (우선순위: ⭐⭐⭐)

**문제**:
- ✅ Gemini 무료 API 사용 (분당 60회, 일 1,500회)
- ❌ **Rate Limit 초과 시 처리 로직 없음**
- ❌ 사용자별 할당량 관리 없음
- ❌ 캐싱 전략 부재 → 동일 종목 반복 조회 시 API 낭비

**영향**:
- Rate Limit 초과 시 서비스 중단
- 일일 1,500회 소진 후 모든 사용자 AI 분석 불가
- 동일 종목 여러 사용자 조회 시 중복 API 호출

**해결 방안**:

#### Option A: Rate Limiter + 캐싱 레이어

```python
# src/infrastructure/external/gemini_client.py (개선)
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional
import hashlib

class GeminiClient(ILLMClient):
    """Google Gemini API 클라이언트 (Rate Limiting + Caching)"""

    def __init__(
        self,
        api_key: str,
        rate_limit_per_minute: int = 60,
        rate_limit_per_day: int = 1500
    ):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Rate Limiting
        self.rpm_limit = rate_limit_per_minute
        self.daily_limit = rate_limit_per_day
        self.request_timestamps = deque()  # 최근 요청 타임스탬프
        self.daily_count = 0
        self.daily_reset_time = datetime.now() + timedelta(days=1)

        # Caching (메모리 기반, 추후 Redis로 전환 가능)
        self._cache = {}  # {prompt_hash: (response, timestamp)}
        self._cache_ttl = 3600  # 1시간

    def generate(
        self,
        prompt: str,
        system_instruction: str = None,
        use_cache: bool = True
    ) -> str:
        """LLM 생성 (Rate Limiting + Caching)"""

        # 1. 캐시 확인
        if use_cache:
            cached = self._get_from_cache(prompt)
            if cached:
                logger.info(f"[Gemini] Cache hit for prompt hash {self._hash_prompt(prompt)[:8]}")
                return cached

        # 2. Rate Limit 체크
        self._check_rate_limit()

        # 3. API 호출
        try:
            response = self.model.generate_content(prompt)
            result = response.text

            # 4. 캐시 저장
            if use_cache:
                self._save_to_cache(prompt, result)

            # 5. Rate Limit 카운터 업데이트
            self._update_rate_limit()

            return result

        except Exception as e:
            logger.error(f"[Gemini] API Error: {e}")
            raise

    def _check_rate_limit(self):
        """Rate Limit 체크 및 대기"""
        now = datetime.now()

        # 일일 한도 리셋
        if now > self.daily_reset_time:
            self.daily_count = 0
            self.daily_reset_time = now + timedelta(days=1)

        # 일일 한도 확인
        if self.daily_count >= self.daily_limit:
            raise RateLimitError(f"Daily limit reached ({self.daily_limit} requests/day)")

        # 분당 한도 확인
        one_minute_ago = now - timedelta(seconds=60)

        # 1분 이내 요청만 유지
        while self.request_timestamps and self.request_timestamps[0] < one_minute_ago:
            self.request_timestamps.popleft()

        # 분당 한도 초과 시 대기
        if len(self.request_timestamps) >= self.rpm_limit:
            oldest_request = self.request_timestamps[0]
            wait_seconds = 60 - (now - oldest_request).seconds

            logger.warning(f"[Gemini] Rate limit reached. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds + 1)

    def _update_rate_limit(self):
        """Rate Limit 카운터 업데이트"""
        self.request_timestamps.append(datetime.now())
        self.daily_count += 1

    def _hash_prompt(self, prompt: str) -> str:
        """프롬프트 해시 생성 (캐시 키)"""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _get_from_cache(self, prompt: str) -> Optional[str]:
        """캐시에서 조회"""
        prompt_hash = self._hash_prompt(prompt)

        if prompt_hash in self._cache:
            response, cached_time = self._cache[prompt_hash]

            # TTL 확인
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return response
            else:
                # 만료된 캐시 삭제
                del self._cache[prompt_hash]

        return None

    def _save_to_cache(self, prompt: str, response: str):
        """캐시에 저장"""
        prompt_hash = self._hash_prompt(prompt)
        self._cache[prompt_hash] = (response, datetime.now())

        # 캐시 크기 제한 (1000개)
        if len(self._cache) > 1000:
            # 가장 오래된 항목 삭제 (간단 구현)
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]


class RateLimitError(Exception):
    """Rate Limit 초과 에러"""
    pass
```

**추가 필요 작업**:
- Phase A-1: `GeminiClient`에 Rate Limiter 구현
- Phase A-1: 메모리 기반 캐싱 구현
- Phase A-1: UI에 "오늘 남은 AI 분석 횟수" 표시

---

## 🟡 개선 권장 사항

### 9. AI 응답 파싱 로직 구체화 (우선순위: ⭐⭐⭐)

**현재 계획**:
- `_parse_response()` 메서드만 언급, 구현 내용 없음

**개선안**:

```python
# src/services/investment_report_service.py (추가)
import re
from typing import Optional

class InvestmentReportService:
    def _parse_response(self, ticker: str, response: str) -> InvestmentReport:
        """AI 응답 파싱"""
        # 정규식으로 구조화된 데이터 추출
        signal_match = re.search(r'신호:\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)', response)
        confidence_match = re.search(r'신뢰도:\s*(\d+)', response)
        summary_match = re.search(r'요약:\s*(.+?)(?:\n|$)', response, re.DOTALL)

        # 파싱 실패 시 예외
        if not signal_match or not confidence_match or not summary_match:
            raise ParsingError(f"Failed to parse AI response: {response[:100]}")

        signal_str = signal_match.group(1)
        confidence = int(confidence_match.group(1))
        summary = summary_match.group(1).strip()

        # Enum 변환
        signal = SignalType[signal_str]

        return InvestmentReport(
            ticker=ticker,
            stock_name=self._get_stock_name(ticker),
            signal=signal,
            confidence_score=confidence,
            summary=summary,
            reasoning=response,  # 전체 응답을 상세 논리로
            generated_at=datetime.now()
        )


class ParsingError(Exception):
    """AI 응답 파싱 실패"""
    pass
```

---

### 10. UI/UX 개선 사항 (우선순위: ⭐⭐)

**현재 계획**:
- 단순 버튼 + 카드 형태만 명시

**개선안**:

```python
# src/dashboard/views/ai_analysis_view.py (개선)
import streamlit as st
import plotly.graph_objects as go

def _display_report(report: InvestmentReport):
    """리포트 카드 UI (개선)"""

    # 신호별 색상 및 이모지
    signal_config = {
        SignalType.STRONG_BUY: {"color": "#2E7D32", "emoji": "🚀", "label": "강력 매수"},
        SignalType.BUY: {"color": "#66BB6A", "emoji": "📈", "label": "매수"},
        SignalType.HOLD: {"color": "#757575", "emoji": "⏸️", "label": "보유"},
        SignalType.SELL: {"color": "#EF5350", "emoji": "📉", "label": "매도"},
        SignalType.STRONG_SELL: {"color": "#D32F2F", "emoji": "💥", "label": "강력 매도"}
    }

    config = signal_config[report.signal]

    # 헤더
    st.markdown(f"### {config['emoji']} {config['label']}")

    # 신뢰도 게이지 (Plotly)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=report.confidence_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "AI 신뢰도"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': config['color']},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "lightyellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=200)
    st.plotly_chart(fig, use_container_width=True)

    # 요약
    st.info(f"📝 **요약**: {report.summary}")

    # 상세 논리 (접기 가능)
    with st.expander("🔍 상세 분석"):
        st.markdown(report.reasoning)

    # 실행 가능 신호 경고
    if report.is_actionable:
        st.success("✅ 실행 가능한 신호입니다. (신뢰도 80% 이상)")
    else:
        st.warning("⚠️ 참고용 신호입니다. 추가 검토가 필요합니다.")
```

---

### 11. 테스트 전략 구체화 (우선순위: ⭐⭐)

**현재 계획**:
- 단순 import 테스트만 명시

**개선안**:

```bash
# tests/unit/test_investment_report_service.py (NEW)
import pytest
from unittest.mock import Mock
from src.services.investment_report_service import InvestmentReportService
from src.infrastructure.external.gemini_client import ILLMClient

def test_generate_report_with_mock_llm():
    """Mock LLM으로 리포트 생성 테스트"""
    # Mock LLM 클라이언트
    mock_llm = Mock(spec=ILLMClient)
    mock_llm.generate.return_value = """
신호: BUY
신뢰도: 75
요약: 기술적 지표 양호
"""

    service = InvestmentReportService(llm_client=mock_llm, stock_repo=Mock())
    report = service.generate_report("005930.KS")

    assert report.signal == SignalType.BUY
    assert report.confidence_score == 75
    assert "기술적" in report.summary


def test_rate_limit_error_handling():
    """Rate Limit 초과 시 에러 처리"""
    from src.infrastructure.external.gemini_client import RateLimitError

    mock_llm = Mock(spec=ILLMClient)
    mock_llm.generate.side_effect = RateLimitError("Daily limit reached")

    service = InvestmentReportService(llm_client=mock_llm, stock_repo=Mock())

    with pytest.raises(RateLimitError):
        service.generate_report("005930.KS")
```

---

## 📊 수정된 구현 일정

### 원래 일정: 10-12일
### 수정 일정: **15-18일** (+50%)

| Phase | 작업 내용 | 원래 | 수정 | 변경 사유 |
|-------|----------|------|------|----------|
| **Phase A-1** | Gemini 클라이언트 + **Rate Limiting** | 1일 | **2일** | Rate Limiter, 캐싱 레이어 추가 |
| **Phase A-2** | 리포트 서비스 + **Phase 20/21 통합** | 2일 | **4일** | 프로필 기반 개인화, Buzz 연동, 프롬프트 템플릿 |
| **Phase B-1** | pykrx + **US 데이터 게이트웨이** | 2일 | **3일** | USDataGateway 추가 구현 |
| **Phase B-2** | 신호 생성 로직 + **가중치 시스템** | 2일 | **3일** | 가중치 기반 점수 계산, 시장 상황 판별 |
| **Phase C-1** | 스크리너 서비스 | 2일 | **2일** | - |
| **Phase C-2** | RecommendationService 통합 | 1일 | **2일** | AI + Traditional 결합 로직 |
| **Phase D (NEW)** | **테스트 작성** | - | **2일** | 단위 테스트, 통합 테스트 |

**총 소요 기간**: 15-18일

---

## 🧪 강화된 검증 계획

### Level 1: 단위 테스트 (추가)

```bash
# Gemini 클라이언트 캐싱 테스트
pytest tests/unit/test_gemini_client.py::test_cache_hit

# 프롬프트 템플릿 테스트
pytest tests/unit/test_prompt_templates.py

# 신호 생성 가중치 테스트
pytest tests/unit/test_signal_generator.py::test_weighted_score
```

### Level 2: 통합 테스트 (추가)

```bash
# Phase 20 프로필 통합 테스트
pytest tests/integration/test_profile_ai_integration.py

# Phase 21 Buzz 통합 테스트
pytest tests/integration/test_buzz_ai_integration.py

# E2E: 사용자 → AI 분석 → 신호 생성
pytest tests/e2e/test_ai_workflow.py
```

---

## 🚀 프로덕션 체크리스트 (추가)

### 배포 전 필수 확인 사항

- [ ] **Phase 20 통합**
  - [ ] 프로필 기반 프롬프트 조정 동작 확인
  - [ ] 안정형 투자자 고변동성 종목 경고 확인
  - [ ] 프로필 없는 사용자 Fallback 동작 확인

- [ ] **Phase 21 통합**
  - [ ] Buzz 데이터가 AI 프롬프트에 포함되는지 확인
  - [ ] 거래량 급증 종목 AI 인지 확인
  - [ ] Screener가 Buzz 점수 고려하는지 확인

- [ ] **Rate Limiting**
  - [ ] 분당 60회 제한 동작 확인
  - [ ] 일일 1,500회 제한 동작 확인
  - [ ] Rate Limit 초과 시 대기 로직 확인

- [ ] **캐싱**
  - [ ] 동일 종목 재조회 시 캐시 적중 확인
  - [ ] 1시간 TTL 만료 후 재조회 확인
  - [ ] 캐시 크기 제한 (1000개) 동작 확인

- [ ] **시장 구분**
  - [ ] 한국 주식: pykrx 데이터 조회 확인
  - [ ] 미국 주식: yfinance 데이터 조회 확인
  - [ ] 시장 자동 감지 정확도 확인

- [ ] **UI/UX**
  - [ ] 신뢰도 게이지 차트 표시 확인
  - [ ] 실행 가능 신호 경고 확인
  - [ ] 상세 분석 접기/펼치기 동작 확인

---

## 📌 최종 권장 사항

### 우선순위 P0 (즉시 반영, Phase A 전)
1. ✅ **Phase 20 프로필 연동** → `InvestmentReportService`에 `profile_repo` 주입
2. ✅ **Phase 21 Buzz 연동** → `market_buzz_service` 주입, 프롬프트에 Buzz 데이터 포함
3. ✅ **기존 SentimentAnalysisService 재사용** → 중복 구현 제거
4. ✅ **Rate Limiting + 캐싱** → `GeminiClient`에 구현

### 우선순위 P1 (Phase B 전까지)
5. ✅ **시장 구분 처리** → `_detect_market()`, `_build_kr_prompt()`, `_build_us_prompt()`
6. ✅ **신호 생성 가중치 시스템** → `SignalGeneratorService` 개선
7. ✅ **프롬프트 템플릿 관리** → `PromptTemplateManager` 구현

### 우선순위 P2 (Phase C 이후)
8. ✅ **RecommendationService 통합** → AI + Traditional 결합
9. ✅ **UI/UX 개선** → Plotly 게이지, 상세 분석 접기
10. ✅ **테스트 작성** → 단위 테스트, 통합 테스트, E2E

---

## 🎯 결론

**강점**:
- ✅ Clean Architecture 완벽 준수
- ✅ 단계적 구현 계획 합리적
- ✅ Zero Cost 전략 (Gemini 무료 API)

**개선 필요**:
- 🔴 **Phase 20 프로필 연동 추가** (프롬프트 개인화, 신호 조정)
- 🔴 **Phase 21 Buzz 연동 추가** (AI 프롬프트에 시장 관심도 반영)
- 🔴 **기존 SentimentAnalysisService 재사용** (중복 개발 방지)
- 🔴 **시장 구분 처리** (한국/미국 데이터 소스 분리)
- 🟡 **Rate Limiting + 캐싱** (API 비용 관리)
- 🟡 **신호 생성 로직 개선** (가중치 기반 점수 계산)
- 🟡 **프롬프트 엔지니어링** (템플릿 관리, Few-shot Learning)

**수정 후 예상 효과**:
- Phase 20 프로필 시스템과 완벽 통합 → 개인화된 AI 추천
- Phase 21 Buzz 시스템 연동 → 시장 관심도 반영한 분석
- 기존 인프라 재사용 → 개발 시간 단축
- Rate Limiting + 캐싱 → 안정적인 서비스 운영
- 가중치 기반 신호 생성 → 허위 신호 감소

**프로덕션 준비도**: 75% → **95%** (수정 후)
- Phase A-B-C 완료 시 즉시 배포 가능
- 테스트는 선택 사항 (수동 테스트 가능하나 권장)

---

**검토 완료일**: 2025-12-25
**다음 단계**: Phase A-1 착수 전 Phase 20/21 통합 설계 검토 및 승인
