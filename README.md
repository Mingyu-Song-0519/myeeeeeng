# 🚀 Intelligent Investment Platform
# 스마트 투자 분석 플랫폼 v2.0

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI 기반 주식 분석, 예측, 포트폴리오 최적화를 위한 종합 투자 대시보드

## 📋 목차
- [✨ 주요 기능](#-주요-기능)
- [🏗️ 아키텍처](#️-아키텍처)
- [🛠️ 설치 방법](#️-설치-방법)
- [🚀 실행 방법](#-실행-방법)
- [📂 프로젝트 구조](#-프로젝트-구조)
- [🎯 기능 상세](#-기능-상세)

---

## ✨ 주요 기능

### 📊 시장 분석
| 기능 | 설명 |
|------|------|
| **실시간 시세** | 한국/미국 시장 실시간 주가 조회 |
| **단일 종목 분석** | 기술적 지표, 차트 패턴, 지지/저항선 분석 |
| **다중 종목 비교** | 여러 종목 수익률/변동성 비교 |
| **시장 체력 진단** | 상승/하락 종목 비율, 52주 신고가/신저가 분석 |

### 🔥 Market Buzz (NEW!)
| 기능 | 설명 |
|------|------|
| **섹터 히트맵** | 업종별 등락률 시각화 (Finviz 스타일) |
| **거래량 급증 감지** | 평소 대비 거래량 급등 종목 알림 |
| **Buzz Score** | 거래량+변동성 기반 관심도 점수 (0~100) |
| **투자 성향 연동** | 사용자 성향에 맞는 종목 필터링 |

### 🤖 AI 예측
| 기능 | 설명 |
|------|------|
| **앙상블 모델** | LSTM + XGBoost + Transformer 결합 예측 |
| **점진적 학습** | 신규 데이터로 모델 자동 업데이트 |
| **신뢰도 평가** | 예측 결과 신뢰 구간 제공 |

### 📰 뉴스 감성 분석
| 기능 | 설명 |
|------|------|
| **한국어 분석** | KR-FinBERT 기반 금융 뉴스 감성 분석 |
| **영어 분석** | VADER 기반 영문 뉴스 감성 분석 |
| **예측 통합** | 감성 점수를 AI 예측 모델에 반영 |

### 👤 투자 성향 진단 (Phase 20)
| 기능 | 설명 |
|------|------|
| **심리 테스트** | 10문항 투자 성향 진단 |
| **맞춤 추천** | 성향 기반 종목 순위 제공 |
| **리스크 매칭** | 위험 감수 성향별 포트폴리오 추천 |

### 💼 포트폴리오 & 리스크
| 기능 | 설명 |
|------|------|
| **최적화** | Markowitz 평균-분산 최적 포트폴리오 |
| **VaR/CVaR** | Value at Risk 리스크 분석 |
| **백테스팅** | 과거 데이터 기반 전략 검증 |
| **팩터 투자** | 가치/모멘텀/품질 팩터 분석 |

---

## 🏗️ 아키텍처

### Clean Architecture 적용
```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                     │
│              (Streamlit Views, UI Components)            │
├─────────────────────────────────────────────────────────┤
│                   Application Layer                      │
│        (Services: MarketBuzzService, ProfileService)     │
├─────────────────────────────────────────────────────────┤
│                     Domain Layer                         │
│          (Entities, Value Objects, Interfaces)           │
├─────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                     │
│       (Repositories, External APIs, Caching)             │
└─────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙
- **의존성 역전 원칙 (DIP)**: 상위 계층이 하위 계층에 의존하지 않음
- **단일 책임 원칙 (SRP)**: 각 모듈은 하나의 책임만 담당
- **Graceful Degradation**: API 실패 시 캐시/폴백 데이터 사용
- **캐싱 전략**: 메모리 캐시 + 파일 캐시 + TTL 관리

---

## 🛠️ 설치 방법

### 요구사항
- Python 3.11+
- pip 또는 conda

### 설치
```bash
# 저장소 클론
git clone https://github.com/Mingyu-Song-0519/Intelligent-Investment-Platform.git
cd Intelligent-Investment-Platform

# 의존성 설치
pip install -r requirements.txt
```

### 선택 설치 (GPU 가속)
```bash
# CUDA 지원 TensorFlow (GPU 사용 시)
pip install tensorflow[and-cuda]
```

---

## 🚀 실행 방법

```bash
# 대시보드 실행
streamlit run src/dashboard/app.py

# 또는 루트 경로에서
streamlit run app.py
```

**접속**: http://localhost:8501

---

## 📂 프로젝트 구조

```
Intelligent-Investment-Platform/
├── 📁 src/
│   ├── 📁 domain/                    # 도메인 계층
│   │   ├── 📁 investment_profile/    # 투자 성향 도메인
│   │   │   ├── entities/             # 엔티티 (Assessment, UserProfile)
│   │   │   ├── value_objects/        # 값 객체 (RiskTolerance)
│   │   │   └── repositories/         # 인터페이스
│   │   └── 📁 market_buzz/           # Market Buzz 도메인
│   │       ├── entities/             # BuzzScore, VolumeAnomaly, SectorHeat
│   │       └── value_objects/        # HeatLevel
│   │
│   ├── 📁 infrastructure/            # 인프라 계층
│   │   ├── repositories/             # 구현체
│   │   └── adapters/                 # 레거시 어댑터
│   │
│   ├── 📁 services/                  # 애플리케이션 계층
│   │   ├── market_buzz_service.py    # Buzz Score 계산
│   │   ├── profile_aware_buzz_service.py  # 성향 연동
│   │   ├── incremental_learning_service.py  # 점진적 학습
│   │   └── ...
│   │
│   ├── 📁 dashboard/                 # 프레젠테이션 계층
│   │   ├── app.py                    # 메인 앱
│   │   ├── views/                    # UI 뷰 컴포넌트
│   │   └── control_center.py         # 투자 컨트롤 센터
│   │
│   ├── 📁 models/                    # AI 모델
│   │   ├── predictor.py              # 기본 예측기
│   │   ├── ensemble_predictor.py     # 앙상블 모델
│   │   └── saved_models/             # 학습된 모델 저장
│   │
│   ├── 📁 analyzers/                 # 분석 모듈
│   │   ├── technical_analyzer.py     # 기술적 분석
│   │   ├── sentiment_analyzer.py     # 감성 분석
│   │   └── market_breadth.py         # 시장 폭 분석
│   │
│   └── 📁 collectors/                # 데이터 수집
│       ├── stock_collector.py        # 주가 데이터
│       └── news_collector.py         # 뉴스 데이터
│
├── 📁 tests/                         # 테스트
├── 📁 work_plan/                     # 개발 계획/문서
├── 📄 requirements.txt               # 의존성
├── 📄 config.py                      # 설정
└── 📄 README.md
```

---

## 🎯 기능 상세

### 📊 섹터 히트맵
![Sector Heatmap](docs/heatmap_preview.png)
- 18개 업종 (반도체, 2차전지, 바이오, 자동차 등)
- 등락률 기준 정렬 및 색상 표시
- 클릭 시 상세 정보 표시

### 👤 투자 성향 진단
1. **10문항 심리 테스트** 진행
2. **위험 감수 성향** 계산 (보수적/중립/공격적)
3. **선호 섹터** 분석
4. **맞춤 종목 추천** 제공

### 🤖 AI 예측 파이프라인
```
데이터 수집 → 전처리 → 특성 추출 → 앙상블 예측 → 결과 시각화
     ↓           ↓           ↓             ↓
  yfinance   기술지표    감성점수     LSTM+XGB+TF
```

---

## 🔧 기술 스택

| 분류 | 기술 |
|------|------|
| **Backend** | Python 3.11, Pandas, NumPy |
| **AI/ML** | TensorFlow, XGBoost, Scikit-learn |
| **NLP** | Transformers (KR-FinBERT), VADER |
| **Frontend** | Streamlit, Plotly |
| **Data** | yfinance, FinanceDataReader |
| **Architecture** | Clean Architecture, DDD |

---

## 📝 라이선스

MIT License - 자유롭게 사용 및 수정 가능

---

## 👨‍💻 개발자

**Mingyu Song**
- GitHub: [@Mingyu-Song-0519](https://github.com/Mingyu-Song-0519)

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 라이브러리들의 도움을 받았습니다:
- [Streamlit](https://streamlit.io/) - 웹 대시보드 프레임워크
- [yfinance](https://github.com/ranaroussi/yfinance) - 주가 데이터 수집
- [Plotly](https://plotly.com/) - 인터랙티브 차트
