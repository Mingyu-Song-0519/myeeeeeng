# 📌 관심 종목(Watchlist) 기능 구현 계획

**Status**: 🔄 계획 검토 대기
**Created**: 2025-12-25

---

## 📋 기능 개요

### 목표
사용자가 선택한 관심 종목들을 한 화면에서 모니터링할 수 있는 기능 구현

### 핵심 기능
1. **관심 종목 추가/삭제** - 종목 검색 및 관리
2. **현재가 조회** - 실시간/지연 시세 표시
3. **간단 분석** - 등락률, 기술지표 요약
4. **알림 연동** - 가격 변동 시 알림 (선택)

---

## 🏗️ Clean Architecture 설계

### 레이어 구조
```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│   watchlist_view.py (Streamlit UI)      │
├─────────────────────────────────────────┤
│         Application Layer               │
│   watchlist_service.py                  │
├─────────────────────────────────────────┤
│           Domain Layer                  │
│   entities/watchlist.py                 │
│   repositories/watchlist_repository.py  │
├─────────────────────────────────────────┤
│       Infrastructure Layer              │
│   repositories/sqlite_watchlist_repo.py │
└─────────────────────────────────────────┘
```

---

## 📦 구현 파일 목록

### Domain Layer (도메인)

#### [NEW] `src/domain/watchlist/entities/watchlist.py`
```python
@dataclass
class WatchlistItem:
    id: str
    user_id: str
    ticker: str
    stock_name: str
    added_at: datetime
    notes: Optional[str] = None
    
@dataclass 
class WatchlistSummary:
    item: WatchlistItem
    current_price: float
    change_pct: float
    volume: int
    rsi: Optional[float]
    signal: str  # "매수", "중립", "매도"
```

#### [NEW] `src/domain/watchlist/repositories/interfaces.py`
```python
class WatchlistRepository(ABC):
    @abstractmethod
    def add_item(self, user_id: str, ticker: str, name: str) -> WatchlistItem
    
    @abstractmethod
    def remove_item(self, user_id: str, ticker: str) -> bool
    
    @abstractmethod
    def get_all(self, user_id: str) -> List[WatchlistItem]
    
    @abstractmethod
    def exists(self, user_id: str, ticker: str) -> bool
```

---

### Infrastructure Layer (인프라)

#### [NEW] `src/infrastructure/repositories/watchlist_repository.py`
- SQLite 기반 영속성 구현
- `watchlist` 테이블 생성/관리
- 캐싱 레이어 포함

---

### Application Layer (서비스)

#### [NEW] `src/services/watchlist_service.py`
```python
class WatchlistService:
    def add_to_watchlist(user_id, ticker, name) -> WatchlistItem
    def remove_from_watchlist(user_id, ticker) -> bool
    def get_watchlist_with_prices(user_id) -> List[WatchlistSummary]
    def get_watchlist_analysis(user_id) -> Dict  # 종합 분석
```

---

### Presentation Layer (UI)

#### [NEW] `src/dashboard/views/watchlist_view.py`
- 관심 종목 목록 테이블
- 종목 추가/삭제 UI
- 현재가 및 등락률 표시
- 간단 기술지표 (RSI, MACD 신호)

#### [MODIFY] `src/dashboard/app.py`
- 새 탭 또는 사이드바 위젯 추가: "⭐ 관심 종목"

---

## 🎨 UI 디자인

### 관심 종목 화면 구성

```
┌─────────────────────────────────────────────────────────┐
│ ⭐ 관심 종목 (5개)                    [+ 종목 추가]     │
├─────────────────────────────────────────────────────────┤
│ 종목명      │ 현재가    │ 등락률  │ RSI  │ 신호 │ 삭제 │
├─────────────────────────────────────────────────────────┤
│ 삼성전자    │ 78,500    │ +2.3%   │  45  │ 중립 │  🗑️  │
│ SK하이닉스  │ 195,000   │ -1.2%   │  32  │ 매수 │  🗑️  │
│ NAVER      │ 215,500   │ +0.8%   │  68  │ 중립 │  🗑️  │
│ 카카오      │ 45,200    │ -0.5%   │  28  │ 매수 │  🗑️  │
│ 현대차      │ 245,000   │ +1.5%   │  55  │ 중립 │  🗑️  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 📊 관심 종목 요약                                        │
├─────────────────────────────────────────────────────────┤
│ 📈 상승: 3개  │  📉 하락: 2개  │  📊 전체 평균: +0.58%   │
└─────────────────────────────────────────────────────────┘
```

---

## 📅 구현 단계

### Phase 1: 도메인 레이어 (1일)
- [ ] `WatchlistItem`, `WatchlistSummary` 엔티티 생성
- [ ] `WatchlistRepository` 인터페이스 정의

### Phase 2: 인프라 레이어 (1일)
- [ ] SQLite 테이블 스키마 설계
- [ ] `SqliteWatchlistRepository` 구현

### Phase 3: 서비스 레이어 (1일)
- [ ] `WatchlistService` 구현
- [ ] 주가 조회 및 기술지표 통합

### Phase 4: UI 레이어 (1일)
- [ ] `watchlist_view.py` 생성
- [ ] `app.py`에 탭/위젯 추가
- [ ] 종목 추가 다이얼로그

### Phase 5: 통합 테스트 (0.5일)
- [ ] 전체 흐름 테스트
- [ ] 기존 "관심 종목 추가" 버튼 연동

---

## ⚠️ 고려사항

### 사용자 식별
- 현재: `st.session_state` 기반 임시 ID
- 권장: 이메일 기반 사용자 ID (이미 구현됨)

### 주가 조회 제한
- yfinance 무료 API 사용
- 캐싱으로 호출 횟수 최소화 (5분 TTL)

### 기존 기능 연동
- `ranking_view.py`의 "관심 종목 추가" 버튼 → `WatchlistService.add_to_watchlist()` 호출로 변경

---

## ✅ 승인 요청

위 구현 계획을 검토 후 승인해주세요.
승인 시 Phase 1부터 구현을 시작하겠습니다.

---
---

# 📋 기획서 검토 및 개선 권장사항

> **검토일**: 2025-12-25
> **검토 기준**: Feature Planner Skill + Clean Architecture + Phase 20/21 통합
> **검토자**: Claude Code (Sonnet 4.5)

---

## ✅ 강점 분석

### 1. Clean Architecture 준수 ⭐⭐⭐⭐⭐

**평가**:
- ✅ Domain/Application/Infrastructure/Presentation 4계층 명확히 분리
- ✅ Repository Pattern 적용 (IWatchlistRepository)
- ✅ Entity 설계 적절 (WatchlistItem, WatchlistSummary)
- ✅ 의존성 역전 원칙(DIP) 준수

**기대 효과**:
- 테스트 가능성 향상 (Mock Repository 주입 가능)
- 데이터 소스 교체 용이 (SQLite → PostgreSQL 등)

---

### 2. 기존 인프라 활용 ⭐⭐⭐⭐

**평가**:
- ✅ yfinance 기존 인프라 재사용
- ✅ SQLite 기존 패턴 활용 (Phase 20 ProfileRepository 참조)
- ✅ Streamlit 캐싱 패턴 일관성

---

### 3. 간결한 MVP 범위 ⭐⭐⭐⭐

**평가**:
- ✅ 핵심 기능에 집중 (추가/삭제/조회)
- ✅ 알림 기능을 선택사항으로 명시
- ✅ 4.5일 일정 합리적

---

## 🔴 중대한 누락 사항

### 1. Phase 20 투자 성향 프로필 연동 미정의 (우선순위: ⭐⭐⭐⭐⭐)

**문제**:
- ✅ Watchlist 기본 기능은 정의됨
- ❌ **Phase 20 투자 성향 프로필과의 통합 방안 없음**
- ❌ 관심 종목의 성향 적합도 분석 로직 없음
- ❌ 사용자 성향에 맞는 종목 추천 기능 없음

**영향**:
- Phase 20에서 구축한 투자 성향 프로필이 활용되지 않음
- 단순 종목 목록 관리로 전락 → 차별화 요소 부족
- 사용자 경험 일관성 저하

**해결 방안**:

#### Option A: WatchlistSummary에 Profile Fit 추가 (권장)

```python
# src/domain/watchlist/entities/watchlist.py (수정)
@dataclass
class WatchlistSummary:
    item: WatchlistItem
    current_price: float
    change_pct: float
    volume: int
    rsi: Optional[float]
    signal: str  # "매수", "중립", "매도"

    # ===== Phase 20 통합 (NEW) =====
    profile_fit_score: Optional[float] = None  # 투자 성향 적합도 (0~100)
    profile_warning: Optional[str] = None  # 성향 불일치 경고
    # 예: "이 종목은 고변동성이므로 안정형 투자자에게 적합하지 않습니다"
```

#### Option B: WatchlistService에 성향 분석 메서드 추가

```python
# src/services/watchlist_service.py (추가)
class WatchlistService:
    def __init__(
        self,
        watchlist_repo: IWatchlistRepository,
        profile_repo: IProfileRepository,  # ← Phase 20
        stock_collector: StockDataCollector
    ):
        self.watchlist_repo = watchlist_repo
        self.profile_repo = profile_repo
        self.stock_collector = stock_collector

    def get_watchlist_with_profile_analysis(
        self,
        user_id: str
    ) -> List[WatchlistSummary]:
        """
        관심 종목 + 투자 성향 적합도 분석

        로직:
        1. 관심 종목 조회
        2. 사용자 프로필 로드
        3. 각 종목의 변동성, 섹터 분석
        4. 프로필 적합도 점수 계산
        5. 경고 메시지 생성 (성향 불일치 시)
        """
        items = self.watchlist_repo.get_all(user_id)
        profile = self.profile_repo.load(user_id)

        summaries = []
        for item in items:
            # 기본 정보 조회
            price_data = self._get_price_data(item.ticker)

            # Phase 20 통합: 성향 적합도 계산
            if profile:
                fit_score = self._calculate_profile_fit(item.ticker, profile)
                warning = self._generate_profile_warning(item.ticker, profile, fit_score)
            else:
                fit_score = None
                warning = None

            summary = WatchlistSummary(
                item=item,
                current_price=price_data['price'],
                change_pct=price_data['change_pct'],
                volume=price_data['volume'],
                rsi=price_data['rsi'],
                signal=self._generate_signal(price_data),
                profile_fit_score=fit_score,  # ← NEW
                profile_warning=warning  # ← NEW
            )
            summaries.append(summary)

        return summaries

    def _calculate_profile_fit(
        self,
        ticker: str,
        profile: InvestorProfile
    ) -> float:
        """
        Phase 20 프로필 기반 적합도 점수 계산

        요소:
        1. 변동성 적합도 (50점)
        2. 섹터 선호도 (30점)
        3. 위험 감수 레벨 매칭 (20점)
        """
        # 종목 정보 조회
        stock_info = self._get_stock_info(ticker)
        volatility = stock_info.get('volatility', 0.3)
        sector = stock_info.get('sector', 'Unknown')

        score = 0.0

        # 1. 변동성 적합도
        ideal_vol_min, ideal_vol_max = profile.get_ideal_volatility_range()
        if ideal_vol_min <= volatility <= ideal_vol_max:
            score += 50
        else:
            ideal_mid = (ideal_vol_min + ideal_vol_max) / 2
            score += max(0, 50 - abs(volatility - ideal_mid) * 100)

        # 2. 섹터 선호도
        if sector in profile.preferred_sectors:
            score += 30
        else:
            score += 10  # 기본 점수

        # 3. 위험 감수 레벨
        risk_value = profile.risk_tolerance.value
        if risk_value <= 40 and volatility < 0.25:  # 안정형 + 저변동성
            score += 20
        elif risk_value > 60 and volatility > 0.35:  # 공격형 + 고변동성
            score += 20
        else:
            score += 10

        return min(100, score)

    def _generate_profile_warning(
        self,
        ticker: str,
        profile: InvestorProfile,
        fit_score: float
    ) -> Optional[str]:
        """성향 불일치 경고 메시지 생성"""
        if fit_score >= 60:
            return None  # 적합도 높으면 경고 없음

        stock_info = self._get_stock_info(ticker)
        volatility = stock_info.get('volatility', 0.3)
        risk_value = profile.risk_tolerance.value

        # 안정형 투자자 + 고변동성 종목
        if risk_value <= 40 and volatility > 0.35:
            return f"⚠️ 이 종목은 변동성이 높아 {profile.profile_type} 투자자에게 적합하지 않을 수 있습니다."

        # 공격형 투자자 + 저변동성 종목
        if risk_value > 60 and volatility < 0.2:
            return f"💡 이 종목은 안정적이지만 {profile.profile_type}에게는 수익률이 낮을 수 있습니다."

        return None
```

**추가 필요 작업**:
- Phase 1: `WatchlistSummary`에 `profile_fit_score`, `profile_warning` 필드 추가
- Phase 3: `WatchlistService`에 `_calculate_profile_fit()` 메서드 구현
- Phase 4: UI에 성향 적합도 표시 (색상 코드: 초록/노랑/빨강)

---

### 2. Phase 21 Market Buzz 연동 미정의 (우선순위: ⭐⭐⭐⭐)

**문제**:
- ✅ Watchlist 기본 기능은 정의됨
- ❌ **Phase 21 Market Buzz와의 통합 방안 없음**
- ❌ 관심 종목의 Buzz 점수 표시 로직 없음
- ❌ 거래량 급증 알림 연동 없음

**영향**:
- Phase 21에서 구축한 Market Buzz 기능이 활용되지 않음
- 사용자가 관심 종목의 시장 관심도를 파악하기 어려움
- 단순 가격/RSI만 표시 → 차별화 요소 부족

**해결 방안**:

#### Option A: WatchlistSummary에 Buzz 정보 추가

```python
# src/domain/watchlist/entities/watchlist.py (수정)
@dataclass
class WatchlistSummary:
    item: WatchlistItem
    current_price: float
    change_pct: float
    volume: int
    rsi: Optional[float]
    signal: str

    # Phase 20 통합
    profile_fit_score: Optional[float] = None
    profile_warning: Optional[str] = None

    # ===== Phase 21 통합 (NEW) =====
    buzz_score: Optional[float] = None  # Market Buzz 점수 (0~100)
    heat_level: Optional[str] = None  # "HOT" | "WARM" | "COLD"
    volume_anomaly: bool = False  # 거래량 급증 여부
```

#### Option B: WatchlistService에 Buzz 분석 추가

```python
# src/services/watchlist_service.py (추가)
class WatchlistService:
    def __init__(
        self,
        watchlist_repo: IWatchlistRepository,
        profile_repo: IProfileRepository,
        stock_collector: StockDataCollector,
        market_buzz_service: MarketBuzzService  # ← Phase 21
    ):
        self.watchlist_repo = watchlist_repo
        self.profile_repo = profile_repo
        self.stock_collector = stock_collector
        self.market_buzz_service = market_buzz_service

    def get_watchlist_with_buzz(
        self,
        user_id: str
    ) -> List[WatchlistSummary]:
        """
        관심 종목 + Market Buzz 분석

        로직:
        1. 관심 종목 조회
        2. 각 종목의 Buzz 점수 계산
        3. 거래량 급증 감지
        4. Heat Level 판정
        """
        items = self.watchlist_repo.get_all(user_id)

        summaries = []
        for item in items:
            # 기본 정보 조회
            price_data = self._get_price_data(item.ticker)

            # Phase 21 통합: Buzz 분석
            try:
                buzz = self.market_buzz_service.calculate_buzz_score(item.ticker)
                buzz_score = buzz.base_score if buzz else None
                heat_level = buzz.heat_level if buzz else None
            except Exception as e:
                logger.warning(f"Failed to get buzz for {item.ticker}: {e}")
                buzz_score = None
                heat_level = None

            # 거래량 급증 감지
            volume_anomaly = self._check_volume_anomaly(item.ticker)

            summary = WatchlistSummary(
                item=item,
                current_price=price_data['price'],
                change_pct=price_data['change_pct'],
                volume=price_data['volume'],
                rsi=price_data['rsi'],
                signal=self._generate_signal(price_data),
                buzz_score=buzz_score,  # ← NEW
                heat_level=heat_level,  # ← NEW
                volume_anomaly=volume_anomaly  # ← NEW
            )
            summaries.append(summary)

        # Buzz 점수 높은 순으로 정렬 옵션
        summaries.sort(key=lambda x: x.buzz_score or 0, reverse=True)

        return summaries
```

**추가 필요 작업**:
- Phase 1: `WatchlistSummary`에 `buzz_score`, `heat_level`, `volume_anomaly` 필드 추가
- Phase 3: `WatchlistService`에 `MarketBuzzService` 의존성 주입
- Phase 4: UI에 Buzz 뱃지 표시 (🔥 HOT / 🌤️ WARM / ❄️ COLD)

---

### 3. 기존 "관심 종목 추가" 버튼 연동 구체화 부족 (우선순위: ⭐⭐⭐⭐)

**문제**:
- ✅ `ranking_view.py`의 "관심 종목 추가" 버튼 존재 확인
- ❌ **현재는 `process_feedback()`으로 추천 수락만 처리**
- ❌ 실제 Watchlist DB에 저장하는 로직 없음
- ❌ 통합 방안 명시되지 않음

**영향**:
- 사용자가 "관심 종목 추가" 버튼을 눌러도 Watchlist에 나타나지 않음
- 기능 간 연결 끊김 → 사용자 혼란

**해결 방안**:

#### ranking_view.py 수정

```python
# src/dashboard/views/ranking_view.py (수정)
def _show_ranking_table(
    ranked_stocks: List[RankedStock],
    service: RecommendationService,
    user_id: str
):
    """순위 테이블 표시"""
    st.subheader("📋 상세 순위")

    # WatchlistService import 추가
    from src.services.watchlist_service import WatchlistService
    from src.infrastructure.repositories.watchlist_repository import SQLiteWatchlistRepository

    watchlist_service = WatchlistService(
        watchlist_repo=SQLiteWatchlistRepository(),
        profile_repo=service.profile_repo,  # 기존 repo 재사용
        stock_collector=service._stock_ranking_service.collector  # 기존 collector 재사용
    )

    for stock in ranked_stocks:
        with st.expander(f"**{stock.rank}위** {stock.stock_name} ({stock.ticker}) - {stock.composite_score:.1f}점"):
            # ... 기존 코드 ...

            # 피드백 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 관심 종목 추가", key=f"accept_{stock.ticker}", use_container_width=True):
                    # 1. 추천 수락 처리 (기존)
                    recs = service.get_user_recommendations(user_id)
                    for rec in recs:
                        if rec.ticker == stock.ticker:
                            service.process_feedback(user_id, rec.recommendation_id, "accept")
                            break

                    # 2. Watchlist에 추가 (NEW)
                    try:
                        watchlist_service.add_to_watchlist(
                            user_id=user_id,
                            ticker=stock.ticker,
                            name=stock.stock_name
                        )
                        st.success(f"{stock.stock_name}을(를) 관심 종목에 추가했습니다!")
                    except Exception as e:
                        st.warning(f"관심 종목 추가 실패: {e}")
```

**추가 필요 작업**:
- Phase 3: `WatchlistService.add_to_watchlist()` 구현
- Phase 5: `ranking_view.py` 통합 테스트
- Phase 5: 중복 추가 방지 로직 (exists() 체크)

---

### 4. 시장 구분 (KR/US) 처리 미정의 (우선순위: ⭐⭐⭐)

**문제**:
- ✅ 티커 저장은 정의됨
- ❌ **한국/미국 종목 구분 로직 없음**
- ❌ 시장별 필터링 기능 없음

**영향**:
- 한국/미국 종목이 섞여서 표시 → 사용자 혼란
- 시장별 현재가 조회 로직 복잡도 증가

**해결 방안**:

#### WatchlistItem에 market 필드 추가

```python
# src/domain/watchlist/entities/watchlist.py (수정)
@dataclass
class WatchlistItem:
    id: str
    user_id: str
    ticker: str
    stock_name: str
    market: str  # ← NEW: "KR" or "US"
    added_at: datetime
    notes: Optional[str] = None
```

#### WatchlistService에 시장 자동 판별

```python
# src/services/watchlist_service.py (추가)
class WatchlistService:
    def add_to_watchlist(
        self,
        user_id: str,
        ticker: str,
        name: str,
        market: Optional[str] = None  # 명시하지 않으면 자동 판별
    ) -> WatchlistItem:
        """관심 종목 추가"""
        # 시장 자동 판별
        if market is None:
            market = self._detect_market(ticker)

        # 중복 체크
        if self.watchlist_repo.exists(user_id, ticker):
            raise ValueError(f"{name}은(는) 이미 관심 종목에 있습니다.")

        # 추가
        item = self.watchlist_repo.add_item(
            user_id=user_id,
            ticker=ticker,
            name=name,
            market=market  # ← NEW
        )

        return item

    def _detect_market(self, ticker: str) -> str:
        """티커에서 시장 자동 판별"""
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            return 'KR'
        elif '.' not in ticker or ticker.endswith('.US'):
            return 'US'
        else:
            # yfinance로 조회하여 확인
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                exchange = stock.info.get('exchange', '')
                if 'KRX' in exchange or 'KSE' in exchange or 'KOE' in exchange:
                    return 'KR'
                else:
                    return 'US'
            except:
                return 'US'  # 기본값
```

**추가 필요 작업**:
- Phase 1: `WatchlistItem`에 `market` 필드 추가
- Phase 2: SQLite 테이블에 `market` 컬럼 추가
- Phase 4: UI에 시장별 탭 또는 필터 추가

---

### 5. 성능 최적화 전략 부재 (우선순위: ⭐⭐⭐⭐)

**문제**:
- ❌ **다수 종목 동시 조회 시 성능 이슈**
- ❌ yfinance API 호출 병렬화 방안 없음
- ❌ 관심 종목 50개 이상 시 로딩 시간 문제

**영향**:
- 관심 종목 10개 → 약 10초 로딩 시간
- 사용자 경험 저하

**해결 방안**:

#### Option A: 병렬 조회 (권장)

```python
# src/services/watchlist_service.py (개선)
import concurrent.futures

class WatchlistService:
    def get_watchlist_with_prices(
        self,
        user_id: str
    ) -> List[WatchlistSummary]:
        """관심 종목 조회 (병렬 처리)"""
        items = self.watchlist_repo.get_all(user_id)

        # 병렬로 가격 데이터 조회
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_item = {
                executor.submit(self._get_summary, item): item
                for item in items
            }

            summaries = []
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    summary = future.result(timeout=10)
                    summaries.append(summary)
                except Exception as e:
                    item = future_to_item[future]
                    logger.error(f"Failed to get summary for {item.ticker}: {e}")
                    # 에러 시 기본값 반환
                    summaries.append(self._get_fallback_summary(item))

        return summaries
```

#### Option B: 캐싱 강화

```python
# src/services/watchlist_service.py (개선)
class WatchlistService:
    def __init__(self, ...):
        # ...
        self._price_cache = {}  # {ticker: (data, timestamp)}
        self._cache_ttl = 300  # 5분

    def _get_price_data(self, ticker: str) -> dict:
        """가격 데이터 조회 (캐싱)"""
        # 캐시 확인
        if ticker in self._price_cache:
            data, cached_time = self._price_cache[ticker]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return data

        # 실시간 조회
        data = self.stock_collector.get_current_price(ticker)
        self._price_cache[ticker] = (data, datetime.now())

        return data
```

**추가 필요 작업**:
- Phase 3: `ThreadPoolExecutor` 병렬 처리 구현
- Phase 3: 캐싱 레이어 강화
- Phase 5: 성능 테스트 (50개 종목 로딩 시간 < 5초)

---

## 🟡 개선 권장 사항

### 6. UI 시각화 개선 (우선순위: ⭐⭐⭐)

**현재 계획**:
- 단순 테이블 형태

**개선안**:

#### Plotly 차트 추가

```python
# src/dashboard/views/watchlist_view.py (추가)
import plotly.graph_objects as go

def _render_watchlist_chart(summaries: List[WatchlistSummary]):
    """관심 종목 등락률 차트"""

    # 데이터 준비
    names = [s.item.stock_name for s in summaries]
    changes = [s.change_pct for s in summaries]
    colors = ['#4CAF50' if c > 0 else '#F44336' for c in changes]

    # 바 차트
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=changes,
            marker_color=colors,
            text=[f"{c:+.2f}%" for c in changes],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="📊 관심 종목 등락률",
        xaxis_title="종목",
        yaxis_title="등락률 (%)",
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig, key="watchlist_chart", use_container_width=True)
```

---

### 7. 정렬/필터링 옵션 추가 (우선순위: ⭐⭐⭐)

**현재 계획**:
- 정렬 옵션 없음

**개선안**:

```python
# src/dashboard/views/watchlist_view.py (추가)
def render_watchlist_view():
    """관심 종목 뷰"""
    st.subheader("⭐ 관심 종목")

    # 정렬 옵션
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        sort_by = st.selectbox(
            "정렬 기준",
            options=["추가일", "등락률", "Buzz 점수", "성향 적합도"],
            key="watchlist_sort"
        )

    with col2:
        sort_order = st.radio(
            "순서",
            options=["내림차순", "오름차순"],
            horizontal=True,
            key="watchlist_order"
        )

    with col3:
        market_filter = st.selectbox(
            "시장",
            options=["전체", "🇰🇷 한국", "🇺🇸 미국"],
            key="watchlist_market"
        )

    # 데이터 조회 및 정렬
    summaries = service.get_watchlist_with_prices(user_id)

    # 필터링
    if market_filter == "🇰🇷 한국":
        summaries = [s for s in summaries if s.item.market == 'KR']
    elif market_filter == "🇺🇸 미국":
        summaries = [s for s in summaries if s.item.market == 'US']

    # 정렬
    if sort_by == "등락률":
        summaries.sort(key=lambda x: x.change_pct, reverse=(sort_order == "내림차순"))
    elif sort_by == "Buzz 점수":
        summaries.sort(key=lambda x: x.buzz_score or 0, reverse=(sort_order == "내림차순"))
    elif sort_by == "성향 적합도":
        summaries.sort(key=lambda x: x.profile_fit_score or 0, reverse=(sort_order == "내림차순"))
    # ...
```

---

### 8. 알림 기능 구체화 (우선순위: ⭐⭐)

**현재 계획**:
- "선택사항"으로만 명시

**개선안**:

```python
# src/domain/watchlist/entities/watchlist.py (추가)
@dataclass
class PriceAlert:
    """가격 알림 설정"""
    id: str
    watchlist_item_id: str
    alert_type: str  # "target_price", "change_pct"
    target_value: float  # 목표가 또는 변동률
    is_active: bool
    created_at: datetime

# src/services/watchlist_service.py (추가)
class WatchlistService:
    def set_price_alert(
        self,
        user_id: str,
        ticker: str,
        alert_type: str,
        target_value: float
    ) -> PriceAlert:
        """가격 알림 설정"""
        # ...

    def check_alerts(self, user_id: str) -> List[str]:
        """알림 조건 체크 (배치 작업)"""
        # ...
```

---

## 📊 수정된 구현 일정

### 원래 일정: 4.5일
### 수정 일정: **6일** (+33%)

| Phase | 작업 내용 | 원래 | 수정 | 변경 사유 |
|-------|----------|------|------|----------|
| **Phase 1** | Domain Layer + **Phase 20/21 필드** | 1일 | **1.5일** | profile_fit, buzz_score 필드 추가 |
| **Phase 2** | Infrastructure Layer + **market 컬럼** | 1일 | **1일** | - |
| **Phase 3** | Service Layer + **Phase 20/21 통합** | 1일 | **2일** | Profile/Buzz 분석 로직 추가 |
| **Phase 4** | UI Layer + **차트/필터링** | 1일 | **1일** | - |
| **Phase 5** | 통합 테스트 + **성능 테스트** | 0.5일 | **0.5일** | 병렬 조회 성능 검증 |

**총 소요 기간**: 6일

---

## 🧪 강화된 테스트 전략

### Level 1: 단위 테스트 (추가)

```python
# tests/unit/test_watchlist_service.py (NEW)
def test_profile_fit_calculation():
    """Phase 20 성향 적합도 계산 테스트"""
    profile = InvestorProfile(
        user_id="test",
        risk_tolerance=RiskTolerance(30),  # 안정형
        preferred_sectors=["Technology"]
    )

    service = WatchlistService(...)

    # 고변동성 종목 → 낮은 적합도
    fit_score = service._calculate_profile_fit("TSLA", profile)
    assert fit_score < 50

    # 저변동성 + 선호 섹터 → 높은 적합도
    fit_score = service._calculate_profile_fit("AAPL", profile)
    assert fit_score > 70
```

### Level 2: 통합 테스트 (추가)

```python
# tests/integration/test_watchlist_ranking_integration.py (NEW)
def test_ranking_to_watchlist_flow():
    """ranking_view → watchlist 통합 테스트"""
    # 1. 추천 종목 조회
    recs = recommendation_service.generate_recommendations(profile)

    # 2. 관심 종목 추가
    watchlist_service.add_to_watchlist(
        user_id="test",
        ticker=recs[0].ticker,
        name=recs[0].stock_name
    )

    # 3. Watchlist 조회
    watchlist = watchlist_service.get_watchlist_with_prices("test")
    assert len(watchlist) == 1
    assert watchlist[0].item.ticker == recs[0].ticker
```

---

## 🚀 프로덕션 체크리스트 (추가)

### 배포 전 필수 확인 사항

- [ ] **Phase 20 통합**
  - [ ] 성향 적합도 계산 정확도 확인
  - [ ] 경고 메시지 표시 확인
  - [ ] 프로필 없는 사용자 Fallback 동작 확인

- [ ] **Phase 21 통합**
  - [ ] Buzz 점수 표시 확인
  - [ ] 거래량 급증 뱃지 확인
  - [ ] Heat Level 색상 코드 확인

- [ ] **성능**
  - [ ] 10개 종목 로딩 < 3초
  - [ ] 50개 종목 로딩 < 10초
  - [ ] 병렬 조회 정상 동작

- [ ] **기존 기능 연동**
  - [ ] ranking_view "관심 종목 추가" 버튼 동작 확인
  - [ ] 중복 추가 방지 확인
  - [ ] 시장별 필터링 동작 확인

---

## 📌 최종 권장 사항

### 우선순위 P0 (즉시 반영)
1. ✅ **Phase 20 프로필 연동** → `profile_fit_score`, `profile_warning` 추가
2. ✅ **Phase 21 Buzz 연동** → `buzz_score`, `heat_level`, `volume_anomaly` 추가
3. ✅ **ranking_view 통합** → `add_to_watchlist()` 호출 추가
4. ✅ **시장 구분 처리** → `market` 필드 추가

### 우선순위 P1 (Phase 3 전까지)
5. ✅ **성능 최적화** → 병렬 조회 구현
6. ✅ **캐싱 강화** → 5분 TTL

### 우선순위 P2 (Phase 4 이후)
7. ✅ **UI 시각화 개선** → Plotly 차트
8. ✅ **정렬/필터링 옵션** → 다양한 정렬 기준

---

## 🎯 결론

**강점**:
- ✅ Clean Architecture 설계 우수
- ✅ 기존 인프라 재사용 합리적
- ✅ MVP 범위 적절

**개선 필요**:
- 🔴 **Phase 20 투자 성향 연동 추가** (profile_fit_score)
- 🔴 **Phase 21 Market Buzz 연동 추가** (buzz_score, heat_level)
- 🔴 **ranking_view 통합 구체화** (add_to_watchlist 호출)
- 🔴 **시장 구분 처리** (market 필드)
- 🟡 **성능 최적화** (병렬 조회)

**수정 후 예상 효과**:
- Phase 20 프로필 시스템과 완벽 통합 → 개인화된 관심 종목 관리
- Phase 21 Buzz 시스템 연동 → 시장 관심도 실시간 파악
- ranking_view와 seamless 연동 → 사용자 경험 일관성
- 병렬 조회로 성능 개선 → 로딩 시간 50% 단축

---

**검토 완료일**: 2025-12-25
**다음 단계**: Phase 1 착수 전 Phase 20/21 통합 설계 검토
