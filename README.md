# EMR 자동 텍스트 입력 도우미

병원 EMR(전자의무기록) 시스템에서 화면을 OCR로 인식하여 환자 정보를 자동으로 추출하고, 전공의에게 보낼 메시지를 자동으로 생성/입력하는 프로그램입니다.

## 주요 기능

- **OCR 자동 인식**: 화면에서 환자 등록번호, 이름, 치료 정보 등을 자동으로 추출
- **메시지 자동 생성**: 추출한 정보를 바탕으로 공손한 메시지 자동 작성
- **단축키 입력**: 한 번의 단축키로 정보 추출부터 메시지 입력까지 자동화
- **한글 완벽 지원**: EasyOCR을 사용한 한글 인식

---

## 🚀 빠른 시작 (사용자용)

### Python 설치 없이 바로 사용

**배포된 실행 파일을 받으셨다면:**

1. `EMR_Helper` 폴더의 `EMR_Helper.exe` 또는 `실행.bat` 더블클릭
2. EMR 화면에서 `Ctrl+Shift+A` 단축키 사용
3. 자동으로 메시지 생성 및 입력!

자세한 사용법은 [`USAGE_GUIDE.md`](USAGE_GUIDE.md)를 참조하세요.

---

## 🛠️ 개발자용 (소스코드 빌드)

### 1. 개발 환경 설정

**필수 요구사항:**
- Python 3.8 이상
- Windows OS

**패키지 설치:**
```bash
# 간편 설치
install.bat

# 또는 수동 설치
pip install -r requirements.txt
```

### 2. 개발 모드로 실행

```bash
python main.py
```

### 3. 실행 파일 빌드

**포터블 버전 빌드 (권장):**
```bash
build_portable.bat
```

빌드 완료 후 `dist\EMR_Helper` 폴더가 생성됩니다.
이 폴더를 배포하면 Python 없이 사용 가능합니다.

**단일 파일 빌드:**
```bash
build.bat
```

단일 .exe 파일로 빌드됩니다. (실행이 느릴 수 있음)

---

## 📖 사용 방법

### 기본 사용

1. EMR 프로그램에서 환자 치료 화면을 엽니다
2. 화면에 다음 정보가 표시되어 있는지 확인:
   - 환자 등록번호 (8자리)
   - 환자 이름
   - 치료실 (예: 2TR)
   - 팀 (예: Z팀)
   - 치료 부위 (예: RLL)
   - 선량/Fx (예: 39.6Gy/18fx)
   - Image Guide (예: G1->G2)
   - Gating (예: No Gating)
3. `Ctrl+Shift+A` 단축키를 누릅니다
4. 프로그램이 자동으로:
   - 화면을 캡처
   - OCR로 정보 추출
   - 메시지 생성
   - 자동 입력 또는 클립보드 복사

### 생성되는 메시지 예시

```
안녕하세요 선생님~
2치료실 Z팀 조병헌님 71046541 RLL RF 39.6Gy/18Fx G1->G2 NG
첫 치료 영상 저장했습니다.
Offline review 확인 부탁드립니다~
```

---

## ⚙️ 설정

`config.json` 파일을 수정하여 설정 변경:

```json
{
  "ocr_mode": true,
  "hotkey": "ctrl+shift+a",
  "auto_type_after_extraction": true,
  "debug_mode": false
}
```

### 주요 설정

- **ocr_mode**: OCR 자동 추출 활성화 여부
  - `true`: 화면에서 자동 추출 (기본값)
  - `false`: 미리 설정된 메시지만 입력

- **hotkey**: 단축키 변경
  - 예: `ctrl+shift+a`, `alt+z`, `ctrl+alt+q`

- **auto_type_after_extraction**: 메시지 자동 입력 여부
  - `true`: 자동 입력
  - `false`: 클립보드에만 복사

- **debug_mode**: 디버그 정보 출력
  - `true`: 상세 정보 출력
  - `false`: 요약 정보만 출력 (기본값)

---

## 📊 추출되는 정보

1. **환자 등록번호**: 8자리 숫자 (예: 71046541)
2. **환자 이름**: 한글 이름 (예: 조병헌)
3. **치료실**: 2TR → "2치료실"
4. **팀**: A~Z팀 (예: Z팀)
5. **치료 부위**: RLL, LLL 등
6. **선량**: 예) 39.6Gy
7. **Fraction 수**: 예) 18Fx
8. **RT Method**: SBRT인 경우만 표시
9. **RF 여부**: FINAL-RF 패턴 감지
10. **Image Guide**: G1->G2 등
11. **Gating**: No Gating(NG), No gating with monitoring(NGM) 등

---

## 📁 프로젝트 구조

```
asanhelper/
├── main.py                 # 메인 프로그램
├── screen_capture.py       # 화면 캡처 모듈
├── ocr_extractor.py        # OCR 정보 추출 모듈
├── message_generator.py    # 메시지 생성 모듈
├── test_ocr.py             # OCR 테스트 스크립트
├── config.json             # 설정 파일
├── requirements.txt        # 필요 패키지
├── README.md               # 개발자 문서
├── USAGE_GUIDE.md          # 사용자 가이드
├── install.bat             # 패키지 설치 스크립트
├── run.bat                 # 개발 모드 실행
├── build.bat               # 단일 파일 빌드
├── build_portable.bat      # 포터블 버전 빌드 (권장)
└── images/                 # EMR 템플릿 이미지
```

---

## 🧪 테스트

### OCR 기능 테스트

```bash
# 기본 테스트 이미지로 테스트
python test_ocr.py

# 특정 이미지로 테스트
python test_ocr.py images/KakaoTalk_20251211_124226257_02.jpg
```

---

## 🔧 트러블슈팅

### OCR이 정보를 제대로 인식하지 못할 때

1. **화면 해상도 확인**: 100~150% 배율 권장
2. **디버그 모드 활성화**: `config.json`에서 `debug_mode: true`
3. **추출 로직 조정**: `ocr_extractor.py`의 정규식 패턴 수정

### 프로그램이 느릴 때

- 첫 실행 시 EasyOCR 모델 로딩으로 느릴 수 있습니다
- 두 번째 실행부터는 빨라집니다
- OCR 처리는 5~10초 소요

### 관리자 권한 필요 시

EMR이 관리자 권한으로 실행되면 이 프로그램도 관리자 권한 필요:

```bash
# 명령 프롬프트를 관리자 권한으로 실행 후
python main.py
```

### GPU 사용 (선택사항)

GPU를 사용하려면:
1. CUDA 설치
2. `ocr_extractor.py`에서 `gpu=False`를 `gpu=True`로 변경

---

## ⚠️ 주의사항

1. **개인정보 보호**: 환자 정보를 다루므로 보안에 주의하세요
2. **정확성 확인**: OCR은 100% 정확하지 않으므로 생성된 메시지를 확인 후 전송하세요
3. **병원 규정**: 자동화 도구 사용이 병원 규정에 위배되지 않는지 확인하세요

---

## 📝 라이선스

개인 사용 목적으로 자유롭게 사용 가능합니다.

---

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
