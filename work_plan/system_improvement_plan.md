# 시스템 점검 및 AI 향상 계획

## 점검일: 2025-12-26

---

## 📊 현재 시스템 구조 요약

| 영역 | 구성 요소 | 현황 |
|-----|---------|-----|
| **AI 예측 모델** | LSTM, XGBoost, Transformer, Ensemble | ⚠️ 학습 데이터 부족 |
| **감성 분석** | VADER, TextBlob, KR-FinBert | ⚠️ 뉴스 수집 제한적 |
| **신호 생성** | 가중치 기반 (AI 35%, 감성 25%, 거래량 20%, 수급 20%) | ✅ 기본 동작 |
| **LLM 통합** | Gemini 2.0-flash, Phase D/E 챗봇 | ✅ 동작 중 |
| **데이터 수집** | Yahoo Finance, pykrx | ⚠️ 실시간 제한 |

---

## 🔴 P0: 즉시 개선 필요 (AI 정확도 직결)

### 1. 데이터 파이프라인 강화

**현재 문제**: Yahoo Finance만 사용 → 데이터 지연/갭 발생

**개선안**:
```python
# src/collectors/__init__.py 확장
class MultiSourceCollector:
    """다중 데이터 소스 수집기"""
    sources = [
        YahooFinanceCollector(),   # 기존
        NaverFinanceCollector(),   # 신규: 한국 주식 실시간
        KISAPICollector(),         # 신규: 한국투자증권 API (실시간)
        AlphaVantageCollector(),   # 신규: 미국 주식 보조
    ]
    
    def fetch_with_fallback(self, ticker: str) -> pd.DataFrame:
        """주 소스 실패 시 대체 소스 사용"""
        for source in self.sources:
            try:
                return source.fetch(ticker)
            except Exception:
                continue
```

**예상 효과**: 데이터 결손율 90% 감소

---

### 2. AI 예측 모델 개선

**현재 문제**: LSTM이 학습 데이터 부족으로 정확도 낮음

**개선안**:

#### 2.1 특성 엔지니어링 강화
```python
# src/models/feature_engineer.py (신규)
class AdvancedFeatureEngineer:
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 기술적 지표 추가
        df['rsi_14'] = ta.RSI(df['close'], 14)
        df['macd'] = ta.MACD(df['close'])['MACD']
        df['bb_width'] = ta.BBANDS(df['close'])['bandwidth']
        
        # 변동성 지표
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'])
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # 모멘텀 지표  
        df['momentum_10'] = df['close'].pct_change(10)
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 시간 특성
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        return df
```

#### 2.2 앙상블 가중치 자동 조정
```python
# src/models/ensemble_predictor.py 수정
class EnsemblePredictor:
    def auto_adjust_weights(self, validation_results: Dict):
        """최근 예측 성능 기반 가중치 동적 조정"""
        total_score = sum(validation_results.values())
        self.weights = {
            model: score / total_score 
            for model, score in validation_results.items()
        }
```

---

### 3. 감성 분석 정확도 향상

**현재 문제**: 키워드 기반 분석은 뉘앙스 파악 불가

**개선안**:
```python
# src/analyzers/sentiment_analyzer.py 수정
class SentimentAnalyzer:
    def __init__(self):
        # Gemini를 감성 분석에도 활용
        self.llm_analyzer = GeminiClient()
        
    def analyze_with_llm(self, text: str) -> Dict:
        """LLM 기반 정밀 감성 분석"""
        prompt = f"""
        다음 금융 뉴스의 감성을 분석하세요:
        "{text}"
        
        JSON 형식으로 답변:
        {{"score": -1.0~1.0, "confidence": 0~1, "keywords": ["..."]}}
        """
        response = self.llm_analyzer.generate(prompt)
        return json.loads(response)
```

---

## 🟡 P1: 중요 개선 (사용자 경험)

### 4. 챗봇 컨텍스트 확장

**현재**: 현재 탭 정보만 전달 → 과거 분석 결과 활용 불가

**개선안**:
```python
# src/domain/chat/entities.py 수정
@dataclass
class ContextData:
    # 기존 필드...
    
    # 신규: 과거 분석 이력
    recent_reports: List[str] = field(default_factory=list)  # 최근 5개 리포트 요약
    recent_signals: List[Dict] = field(default_factory=list)  # 최근 신호
    watchlist_tickers: List[str] = field(default_factory=list)  # 관심 종목
```

### 5. 프롬프트 엔지니어링 개선

**현재**: 단순 지시형 프롬프트

**개선안** (`context_assembler.py`):
```python
# Few-shot 예시 추가
ANALYST_EXAMPLES = """
예시 1:
Q: "삼성전자 지금 살까요?"
A: "현재 RSI가 28로 과매도 구간입니다. 기관이 3일 연속 순매수 중이며, 
    PBR 1.2는 역사적 저점 대비 매력적입니다. 단기 반등 가능성 높습니다.
    다만, 반도체 업황 둔화 리스크가 있어 분할 매수를 권장합니다."

예시 2:
Q: "오늘 시장 어때?"
A: "KOSPI가 -1.2% 하락한 가운데, 외국인 5,000억 순매도가 부담입니다.
    반도체, 자동차 약세, 방산/조선 강세입니다. 
    현금 비중 확대를 고려할 시점입니다."
"""
```

---

## 🟢 P2: 추가 개선 (고급 기능)

### 6. 자동 백테스팅 파이프라인

AI 신호의 과거 성과 자동 측정 → 모델 개선 피드백

### 7. 알림 서비스 연동

신호 발생 시 Telegram/Slack 알림

### 8. 다중 시장 확장

미국 시장 데이터 수집 및 분석 강화

---

## 📋 구현 우선순위 요약

| 순위 | 작업 | 예상 시간 | 영향도 |
|-----|-----|---------|-------|
| P0-1 | 특성 엔지니어링 강화 | 2-3시간 | 🔥🔥🔥 |
| P0-2 | 감성 분석 LLM 활용 | 2시간 | 🔥🔥🔥 |
| P0-3 | 데이터 소스 다변화 | 3-4시간 | 🔥🔥 |
| P1-1 | 챗봇 컨텍스트 확장 | 2시간 | 🔥🔥 |
| P1-2 | 프롬프트 개선 | 1시간 | 🔥🔥 |

---

## ✅ 다음 단계

1. 위 계획 검토 후 승인
2. P0-1 (특성 엔지니어링) 구현 시작
3. 순차적 개선 진행

---

## 🔍 Feature Planner 검토 및 개선 권장사항

**검토일**: 2025-12-26
**검토 기준**: Clean Architecture 준수, TDD 방법론, Phase A/B/C/D/E 통합, 확장성

### 1. Clean Architecture 검토 결과

#### ✅ 현재 구조 분석

**기존 파일 구조 확인**:
- ✅ **Infrastructure Layer**: `src/collectors/` (데이터 수집), `src/infrastructure/external/`
- ✅ **Application Layer**: `src/services/`, `src/models/` (비즈니스 로직)
- ✅ **Domain Layer**: `src/domain/` (엔티티, VO)
- ✅ **Presentation Layer**: `src/dashboard/` (UI)

**발견된 아키텍처 이슈**:

##### 1.1 계층 분리 위반: `src/models/` 위치 모호성
**문제점**:
- `EnsemblePredictor`, `LSTMPredictor` 등이 `src/models/`에 위치
- 이는 **Application Layer 서비스인지** **Domain Layer 로직인지** 불명확

**권장 수정**:
```
현재 구조:
src/
  models/
    ensemble_predictor.py    # ❌ 모호한 위치
    predictor.py

권장 구조:
src/
  domain/
    prediction/
      entities.py            # PredictionResult, ModelMetrics 등
      value_objects.py       # Confidence, SignalStrength 등
  services/
    prediction/
      ensemble_service.py    # ✅ Application Layer 서비스
      model_trainer.py       # ✅ 학습 로직
  infrastructure/
    ml_models/
      lstm_model.py          # ✅ 실제 ML 모델 구현체
      xgboost_model.py
      transformer_model.py
```

##### 1.2 데이터 수집 인터페이스 부재 (DIP 위반)
**문제점**:
- `StockDataCollector`가 직접 yfinance에 의존
- 다른 데이터 소스 추가 시 코드 수정 필요

**권장 해결책**:
```python
# src/domain/market_data/interfaces.py (Domain Layer)
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class IStockDataGateway(ABC):
    """주식 데이터 게이트웨이 인터페이스 (DIP)"""

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str
    ) -> Optional[pd.DataFrame]:
        """OHLCV 데이터 조회"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """데이터 소스 사용 가능 여부"""
        pass

# src/infrastructure/market_data/yahoo_gateway.py
class YahooFinanceGateway(IStockDataGateway):
    def fetch_ohlcv(self, ticker, start, end):
        stock = yf.Ticker(ticker)
        return stock.history(start=start, end=end)

    def is_available(self) -> bool:
        # API 헬스체크
        return True

# src/infrastructure/market_data/naver_gateway.py
class NaverFinanceGateway(IStockDataGateway):
    def fetch_ohlcv(self, ticker, start, end):
        # Naver 크롤링 로직
        return df

    def is_available(self) -> bool:
        return True

# src/infrastructure/market_data/fallback_gateway.py
class FallbackStockDataGateway(IStockDataGateway):
    """Fallback 패턴 구현"""

    def __init__(self, gateways: List[IStockDataGateway]):
        self.gateways = gateways

    def fetch_ohlcv(self, ticker, start, end):
        for gateway in self.gateways:
            if not gateway.is_available():
                continue
            try:
                df = gateway.fetch_ohlcv(ticker, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Gateway {gateway.__class__.__name__} failed: {e}")
        raise DataUnavailableError("All gateways failed")
```

##### 1.3 Feature Engineering 위치 선정
**제안된 위치**: `src/models/feature_engineer.py` (❌)

**권장 위치**: `src/services/feature_engineering_service.py` (✅)

**이유**:
- Feature Engineering은 **비즈니스 로직** (Application Layer)
- Domain 엔티티를 변환하여 ML 모델 입력으로 가공하는 Orchestration 역할

```python
# src/services/feature_engineering_service.py (Application Layer)
from src.domain.market_data.entities import OHLCV
from src.domain.prediction.value_objects import TechnicalFeatures

class FeatureEngineeringService:
    """기술적 지표 및 특성 생성 서비스"""

    def create_technical_features(self, ohlcv: OHLCV) -> TechnicalFeatures:
        """OHLCV 데이터에서 기술적 특성 생성"""
        df = ohlcv.to_dataframe()

        # RSI, MACD 등 계산
        rsi = self._calculate_rsi(df['close'], 14)
        macd = self._calculate_macd(df['close'])

        return TechnicalFeatures(
            rsi=rsi,
            macd=macd,
            # ...
        )
```

---

### 2. 제안된 개선안별 상세 분석

#### P0-1: 특성 엔지니어링 강화

**Clean Architecture 재설계**:

```
┌─────────────────────────────────────────────────────┐
│  Application Layer                                  │
│  - FeatureEngineeringService                        │
│    └─ create_technical_features()                   │
│    └─ create_momentum_features()                    │
│    └─ create_volatility_features()                  │
└──────────────────┬──────────────────────────────────┘
                   │ 의존성 ↓
┌──────────────────▼──────────────────────────────────┐
│  Domain Layer                                       │
│  - TechnicalFeatures (Value Object)                 │
│  - FeatureVector (Entity)                           │
└─────────────────────────────────────────────────────┘
```

**TDD 접근**:

1. **RED Phase**: 테스트 먼저 작성
```python
# tests/services/test_feature_engineering_service.py
class TestFeatureEngineeringService:
    def test_create_rsi_feature_returns_correct_range(self):
        # Given
        service = FeatureEngineeringService()
        sample_data = create_sample_ohlcv(days=30)

        # When
        features = service.create_technical_features(sample_data)

        # Then
        assert 0 <= features.rsi <= 100
        assert features.rsi is not None

    def test_create_macd_feature_with_insufficient_data_returns_none(self):
        # Given
        service = FeatureEngineeringService()
        sample_data = create_sample_ohlcv(days=5)  # 너무 짧음

        # When
        features = service.create_technical_features(sample_data)

        # Then
        assert features.macd is None
```

2. **GREEN Phase**: 최소 구현
```python
class FeatureEngineeringService:
    def create_technical_features(self, ohlcv: OHLCV) -> TechnicalFeatures:
        df = ohlcv.to_dataframe()

        if len(df) < 14:
            return TechnicalFeatures(rsi=None, macd=None)

        rsi = self._calculate_rsi(df['close'], 14)

        if len(df) < 26:
            return TechnicalFeatures(rsi=rsi, macd=None)

        macd = self._calculate_macd(df['close'])

        return TechnicalFeatures(rsi=rsi, macd=macd)
```

3. **REFACTOR Phase**: 코드 개선
- `_calculate_rsi()`, `_calculate_macd()` 메서드 분리
- 매직 넘버 → 상수로 추출 (`RSI_PERIOD = 14`)

**파일 구조**:
```
src/
  domain/
    prediction/
      value_objects.py       # TechnicalFeatures, MomentumFeatures
  services/
    feature_engineering_service.py
  infrastructure/
    technical_indicators/
      rsi_calculator.py      # 순수 계산 로직
      macd_calculator.py
tests/
  services/
    test_feature_engineering_service.py
  infrastructure/
    test_rsi_calculator.py
```

**Coverage Target**: ≥90% (핵심 비즈니스 로직)

---

#### P0-2: 감성 분석 LLM 활용

**Clean Architecture 재설계**:

**문제점**: 원안에서 `SentimentAnalyzer`가 직접 `GeminiClient`를 생성
```python
# ❌ 잘못된 설계
class SentimentAnalyzer:
    def __init__(self):
        self.llm_analyzer = GeminiClient()  # Infrastructure에 직접 의존!
```

**권장 설계**:
```python
# ✅ DIP 준수 설계
# src/domain/sentiment/interfaces.py
class ISentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        pass

# src/services/sentiment_analysis_service.py (Application Layer)
class SentimentAnalysisService:
    def __init__(
        self,
        llm_client: Optional[ILLMClient] = None,
        vader_analyzer: Optional[ISentimentAnalyzer] = None
    ):
        self.llm_analyzer = LLMSentimentAnalyzer(llm_client) if llm_client else None
        self.vader_analyzer = vader_analyzer or VaderSentimentAnalyzer()

    def analyze_news(self, text: str, use_llm: bool = True) -> SentimentResult:
        """뉴스 감성 분석 (LLM 우선, VADER fallback)"""
        if use_llm and self.llm_analyzer:
            try:
                return self.llm_analyzer.analyze(text)
            except Exception as e:
                logger.warning(f"LLM analysis failed, fallback to VADER: {e}")

        return self.vader_analyzer.analyze(text)

# src/infrastructure/sentiment/llm_sentiment_analyzer.py
class LLMSentimentAnalyzer(ISentimentAnalyzer):
    def __init__(self, llm_client: ILLMClient):
        self.llm_client = llm_client

    def analyze(self, text: str) -> SentimentResult:
        prompt = self._build_sentiment_prompt(text)
        response = self.llm_client.generate(prompt)
        return self._parse_response(response)
```

**TDD 전략**:
```python
# tests/services/test_sentiment_analysis_service.py
class TestSentimentAnalysisService:
    def test_analyze_news_uses_llm_when_available(self):
        # Given
        mock_llm = Mock(spec=ILLMClient)
        mock_llm.generate.return_value = '{"score": 0.8, "confidence": 0.9}'

        service = SentimentAnalysisService(llm_client=mock_llm)

        # When
        result = service.analyze_news("Stock prices soar", use_llm=True)

        # Then
        assert result.score == 0.8
        assert mock_llm.generate.called

    def test_analyze_news_falls_back_to_vader_on_llm_failure(self):
        # Given
        mock_llm = Mock(spec=ILLMClient)
        mock_llm.generate.side_effect = Exception("API Error")

        service = SentimentAnalysisService(llm_client=mock_llm)

        # When
        result = service.analyze_news("Stock prices soar", use_llm=True)

        # Then
        assert result is not None  # VADER가 동작함
        assert result.score > 0  # 긍정 감성
```

**Rate Limiting 고려**:
- LLM 감성 분석은 **뉴스 배치 처리 시** 사용 (실시간 X)
- 1일 1회 뉴스 수집 → 배치 감성 분석 → 캐시 저장

---

#### P0-3: 데이터 소스 다변화

**Multi-Source Pattern 구현**:

```python
# src/services/market_data_service.py (Application Layer)
class MarketDataService:
    """시장 데이터 수집 오케스트레이션"""

    def __init__(
        self,
        gateways: List[IStockDataGateway],
        cache_repo: Optional[IMarketDataCache] = None
    ):
        self.fallback_gateway = FallbackStockDataGateway(gateways)
        self.cache_repo = cache_repo

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        use_cache: bool = True
    ) -> OHLCV:
        """캐시 우선, 다중 소스 폴백 데이터 조회"""

        # 1. 캐시 확인
        if use_cache and self.cache_repo:
            cached = self.cache_repo.get(ticker, start, end)
            if cached:
                return cached

        # 2. 다중 소스 폴백
        df = self.fallback_gateway.fetch_ohlcv(ticker, start, end)

        if df is None or df.empty:
            raise DataNotFoundError(f"No data for {ticker}")

        # 3. Domain 엔티티 변환
        ohlcv = OHLCV.from_dataframe(ticker, df)

        # 4. 캐시 저장
        if self.cache_repo:
            self.cache_repo.save(ohlcv)

        return ohlcv
```

**Gateway 우선순위 설정**:
```python
# src/infrastructure/market_data/gateway_factory.py
class GatewayFactory:
    @staticmethod
    def create_gateways(market: str) -> List[IStockDataGateway]:
        """시장별 최적 게이트웨이 생성"""
        if market == "KR":
            return [
                KISAPIGateway(),         # 1순위: 실시간 API
                PyKRXGateway(),          # 2순위: pykrx
                NaverFinanceGateway(),   # 3순위: 크롤링
                YahooFinanceGateway()    # 4순위: Yahoo (보조)
            ]
        else:  # US
            return [
                AlphaVantageGateway(),   # 1순위: Alpha Vantage
                YahooFinanceGateway(),   # 2순위: Yahoo Finance
            ]
```

---

#### P1-1: 챗봇 컨텍스트 확장

**제안된 방식의 문제점**:
```python
# ❌ 메모리 기반 - 재시작 시 손실
@dataclass
class ContextData:
    recent_reports: List[str] = field(default_factory=list)
```

**권장 방식**: Repository 패턴 사용
```python
# src/domain/chat/interfaces.py
class IChatHistoryRepository(ABC):
    @abstractmethod
    def save_report(self, user_id: str, report: InvestmentReport):
        pass

    @abstractmethod
    def get_recent_reports(self, user_id: str, limit: int = 5) -> List[InvestmentReport]:
        pass

# src/infrastructure/repositories/chat_history_repository.py
class SQLiteChatHistoryRepository(IChatHistoryRepository):
    def get_recent_reports(self, user_id: str, limit: int = 5):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ticker, signal, confidence_score, summary, created_at
                FROM chat_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            # ...
            return reports

# src/services/chat/context_assembler.py 수정
class ContextAssembler:
    def __init__(self, history_repo: Optional[IChatHistoryRepository] = None):
        self.history_repo = history_repo

    def assemble_system_prompt(self, context: ContextData) -> str:
        prompt = "당신은 AI 투자 비서입니다.\n\n"

        # 과거 분석 이력 포함
        if self.history_repo:
            recent_reports = self.history_repo.get_recent_reports(
                context.user_id,
                limit=3
            )
            if recent_reports:
                prompt += "**최근 분석 이력**:\n"
                for r in recent_reports:
                    prompt += f"- {r.ticker}: {r.signal.value} (신뢰도 {r.confidence_score})\n"
                prompt += "\n"

        # 현재 화면 컨텍스트
        prompt += f"**현재 화면**: {context.tab_name}\n"
        # ...
```

---

### 3. 구현 Phase 분할 (TDD 기반)

#### Phase 1: Data Gateway 인터페이스 및 Fallback 구현 (4-5시간)

**목표**: 다중 데이터 소스 지원 Infrastructure 구축

**Tasks**:
1. **RED**:
   - `test_fallback_gateway_uses_second_source_on_first_failure()`
   - `test_fallback_gateway_raises_error_when_all_fail()`
   - `test_yahoo_gateway_returns_valid_dataframe()`
2. **GREEN**:
   - `IStockDataGateway` 인터페이스 정의
   - `YahooFinanceGateway`, `NaverFinanceGateway` 구현
   - `FallbackStockDataGateway` 구현
3. **REFACTOR**:
   - Gateway 팩토리 패턴 적용
   - 에러 처리 통일

**파일 생성**:
- `src/domain/market_data/interfaces.py` (NEW)
- `src/infrastructure/market_data/yahoo_gateway.py` (REFACTOR from stock_collector.py)
- `src/infrastructure/market_data/naver_gateway.py` (NEW)
- `src/infrastructure/market_data/fallback_gateway.py` (NEW)
- `tests/infrastructure/market_data/test_fallback_gateway.py` (NEW)

**Quality Gate**:
- [ ] Gateway 인터페이스 테스트 커버리지 100%
- [ ] Fallback 로직 테스트 통과 (3개 소스 순차 시도)
- [ ] 기존 `StockDataCollector` 사용 코드 모두 동작 확인

---

#### Phase 2: Feature Engineering Service 구현 (3-4시간)

**목표**: 고급 기술적 지표 생성 서비스

**Tasks**:
1. **RED**:
   - `test_create_rsi_with_valid_data()`
   - `test_create_macd_with_insufficient_data_returns_none()`
   - `test_create_volatility_features()`
2. **GREEN**:
   - `FeatureEngineeringService` 클래스 구현
   - RSI, MACD, Bollinger Bands, ATR 계산 로직
3. **REFACTOR**:
   - 지표 계산 로직 → Infrastructure Layer로 분리
   - 매직 넘버 → 상수화

**파일 생성**:
- `src/domain/prediction/value_objects.py` (NEW: TechnicalFeatures)
- `src/services/feature_engineering_service.py` (NEW)
- `src/infrastructure/technical_indicators/rsi_calculator.py` (NEW)
- `tests/services/test_feature_engineering_service.py` (NEW)

**Dependencies**:
- Phase 1 완료 필요 (데이터 조회 안정성)

**Quality Gate**:
- [ ] Service 테스트 커버리지 ≥90%
- [ ] 모든 기술적 지표 정확도 검증 (Known Values 테스트)
- [ ] 기존 AI 예측 모델과 통합 테스트

---

#### Phase 3: LLM Sentiment Analyzer 구현 (2-3시간)

**목표**: Gemini 기반 고급 감성 분석

**Tasks**:
1. **RED**:
   - `test_llm_sentiment_returns_score_in_range()`
   - `test_llm_sentiment_fallback_to_vader_on_error()`
   - `test_sentiment_caching_reduces_api_calls()`
2. **GREEN**:
   - `LLMSentimentAnalyzer` 구현
   - Fallback 로직 (LLM → VADER)
   - 감성 점수 캐싱 (Redis 또는 SQLite)
3. **REFACTOR**:
   - 프롬프트 템플릿 분리
   - 응답 파싱 로직 강화

**파일 수정/생성**:
- `src/infrastructure/sentiment/llm_sentiment_analyzer.py` (NEW)
- `src/services/sentiment_analysis_service.py` (MODIFY: DI 추가)
- `tests/services/test_sentiment_analysis_service.py` (MODIFY)

**Dependencies**:
- Phase E (GeminiClient) 필요

**Quality Gate**:
- [ ] LLM 감성 분석 정확도 ≥85% (수동 테스트 20건)
- [ ] Fallback 로직 동작 확인
- [ ] 캐싱으로 API 호출 50% 감소

---

#### Phase 4: Market Data Service 통합 (2-3시간)

**목표**: 다중 소스 통합 및 캐싱

**Tasks**:
1. **RED**:
   - `test_market_data_service_uses_cache_first()`
   - `test_market_data_service_fallback_on_cache_miss()`
2. **GREEN**:
   - `MarketDataService` 구현
   - 캐시 Repository 구현 (SQLite 기반)
3. **REFACTOR**:
   - 기존 collector 사용 코드 → MarketDataService로 교체

**파일 생성**:
- `src/services/market_data_service.py` (NEW)
- `src/infrastructure/repositories/market_data_cache_repository.py` (NEW)
- `tests/services/test_market_data_service.py` (NEW)

**Quality Gate**:
- [ ] 캐시 히트율 ≥70% (100회 조회 테스트)
- [ ] 모든 Phase A/B/C 기능 정상 동작
- [ ] 데이터 결손율 90% 감소 확인

---

#### Phase 5: Chat History Repository 구현 (2시간)

**목표**: 과거 분석 이력 저장 및 조회

**Tasks**:
1. **RED**:
   - `test_save_report_stores_in_database()`
   - `test_get_recent_reports_returns_latest_5()`
2. **GREEN**:
   - `IChatHistoryRepository` 인터페이스
   - `SQLiteChatHistoryRepository` 구현
3. **REFACTOR**:
   - ContextAssembler에 통합

**파일 생성**:
- `src/domain/chat/interfaces.py` (MODIFY: 인터페이스 추가)
- `src/infrastructure/repositories/chat_history_repository.py` (NEW)
- `tests/infrastructure/repositories/test_chat_history_repository.py` (NEW)

**Quality Gate**:
- [ ] Repository 테스트 커버리지 100%
- [ ] 챗봇이 과거 분석 이력 활용 확인 (수동 테스트)

---

#### Phase 6: Ensemble Model Auto-Weight Adjustment (2-3시간)

**목표**: 예측 성능 기반 가중치 자동 조정

**Tasks**:
1. **RED**:
   - `test_auto_adjust_weights_increases_best_model_weight()`
   - `test_ensemble_with_auto_weights_improves_accuracy()`
2. **GREEN**:
   - `EnsemblePredictor.auto_adjust_weights()` 구현
   - 검증 데이터 기반 성능 측정
3. **REFACTOR**:
   - 가중치 조정 알고리즘 개선 (Softmax, EMA 등)

**파일 수정**:
- `src/models/ensemble_predictor.py` (MODIFY: 기존 파일)
- `tests/models/test_ensemble_predictor.py` (NEW)

**Quality Gate**:
- [ ] 자동 가중치 조정으로 앙상블 정확도 5% 향상
- [ ] 가중치 수렴 확인 (10회 조정 후 안정화)

---

### 4. 위험 요소 및 완화 전략

| 위험 | 확률 | 영향 | 완화 전략 |
|------|-----|-----|---------|
| **데이터 소스 API 변경/중단** | 중간 | 높음 | Fallback 패턴, 최소 3개 소스 유지, Gateway 인터페이스로 격리 |
| **LLM API 비용 증가** | 높음 | 중간 | 캐싱 필수, 배치 처리, Rate Limiting (1일 100건 제한) |
| **Feature Engineering으로 인한 학습 시간 증가** | 높음 | 낮음 | Lazy Evaluation, 필수 지표만 우선 계산 |
| **캐시 데이터 일관성 문제** | 낮음 | 중간 | TTL 설정 (1일), 캐시 무효화 로직 |
| **기존 코드와의 통합 오류** | 중간 | 높음 | 단계별 통합, 각 Phase에서 기존 기능 Regression Test |

---

### 5. 최종 권장사항 요약

#### 필수 수정 사항 (P0)

1. **Clean Architecture 재구성**
   - `src/models/` → `src/services/prediction/` + `src/infrastructure/ml_models/`로 분리
   - 모든 외부 의존성에 Interface 추가 (IStockDataGateway, ISentimentAnalyzer)

2. **DIP (의존성 역전 원칙) 철저히 준수**
   - Application Layer가 Infrastructure 구현체에 직접 의존 금지
   - 모든 Service는 생성자에서 Interface 주입받기

3. **Fallback 패턴 필수 구현**
   - 데이터 소스, LLM 감성 분석 모두 Fallback 체인 구축

4. **TDD 방법론 엄수**
   - 모든 신규 기능: RED → GREEN → REFACTOR 순서
   - 테스트 없이 프로덕션 코드 작성 금지

#### 권장 개선 사항 (P1)

1. **캐싱 레이어 추가**
   - 시장 데이터: SQLite 캐시 (TTL 1일)
   - LLM 감성 분석: Redis 캐시 (TTL 7일)

2. **Repository 패턴 적용**
   - ChatHistory, MarketDataCache, PredictionHistory

3. **Monitoring & Logging 강화**
   - 각 Gateway 성공률 추적
   - LLM API 사용량 모니터링

#### 선택 사항 (P2)

1. **Auto-Scaling Gateway Pool**
   - 데이터 소스별 Health Check 주기적 실행
   - 실패율 높은 Gateway 자동 제외

2. **Feature Store 구축**
   - 계산된 기술적 지표 재사용
   - 배치 계산 → 실시간 조회

---

### 6. 구현 순서 최종 확정

**권장 순서** (의존성 고려):
1. **Phase 1**: Data Gateway 인터페이스 (4-5h) → 모든 Phase의 기반
2. **Phase 2**: Feature Engineering Service (3-4h) → AI 모델 정확도 향상 직결
3. **Phase 4**: Market Data Service 통합 (2-3h) → Gateway + Cache 통합
4. **Phase 3**: LLM Sentiment Analyzer (2-3h) → Phase E Gemini 활용
5. **Phase 5**: Chat History Repository (2h) → 챗봇 UX 향상
6. **Phase 6**: Ensemble Auto-Weight (2-3h) → AI 최적화

**총 예상 시간**: 15-20시간 (2-3일 집중 작업)

---

### 7. Rollback 전략

각 Phase별 롤백 방법:

- **Phase 1**: 기존 `StockDataCollector` 유지 → Gateway 미사용 시 영향 없음
- **Phase 2**: `FeatureEngineeringService` 미사용 → 기존 단순 지표 계속 사용
- **Phase 3**: `LLMSentimentAnalyzer` 오류 시 → VADER로 자동 Fallback
- **Phase 4**: `MarketDataService` 문제 시 → 직접 Gateway 호출로 복귀
- **Phase 5**: ChatHistory 오류 시 → ContextAssembler에서 history_repo=None 처리
- **Phase 6**: Auto-Weight 비활성화 → 고정 가중치 사용

---

**검토 완료일**: 2025-12-26
**다음 단계**: 사용자 승인 후 Phase 1부터 TDD 기반 구현 시작
**예상 완료일**: Phase 1-6 완료 시 AI 정확도 30-50% 개선 예상
