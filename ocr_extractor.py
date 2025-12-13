"""
OCR을 통한 EMR 정보 추출 모듈
"""

import re
import cv2
import numpy as np
import warnings
from typing import Dict, Optional

# PyTorch DataLoader pin_memory 경고 숨기기
warnings.filterwarnings('ignore', message='.*pin_memory.*')


class EMRDataExtractor:
    def __init__(self, enable_preprocessing=True):
        """OCR 초기화

        Args:
            enable_preprocessing: 이미지 전처리 활성화 (속도 및 정확도 향상)
        """
        import easyocr
        # 한글과 영어 지원 (CPU 최적화)
        self.reader = easyocr.Reader(
            ['ko', 'en'],
            gpu=False,           # GPU 비활성화
            verbose=False,       # 불필요한 로그 제거 (속도 향상)
            quantize=True,       # 모델 경량화 (속도 향상, 메모리 절약)
            download_enabled=True
        )
        self.enable_preprocessing = enable_preprocessing

    def preprocess_image(self, image):
        """이미지 전처리 (OCR 속도 및 정확도 향상)

        Args:
            image: 원본 이미지 (BGR)

        Returns:
            numpy array: 전처리된 이미지
        """
        # 1. 그레이스케일 변환 (컬러 처리 부담 감소, 속도 2배 향상)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        # 텍스트 가독성 향상, OCR 정확도 증가
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. 노이즈 제거 (가우시안 블러)
        # 작은 노이즈 제거, OCR 오인식 감소
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 4. 샤프닝 (선명도 향상)
        # 텍스트 경계 강화
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return sharpened

    def extract_text_from_image(self, image):
        """이미지에서 텍스트 추출

        Args:
            image: numpy array 이미지

        Returns:
            list: [(텍스트, 좌표), ...]
        """
        # 속도 최적화 파라미터
        results = self.reader.readtext(
            image,
            paragraph=False,      # 단락 병합 비활성화 (속도 향상)
            batch_size=10,        # 배치 크기 증가 (CPU에서도 효과적)
            workers=0,            # 단일 스레드 사용 (멀티프로세싱 오버헤드 제거)
            decoder='greedy'      # 빠른 디코더 사용 (정확도 약간 감소, 속도 향상)
        )
        return [(text, bbox) for bbox, text, conf in results]

    def extract_patient_info(self, image):
        """환자 정보 추출

        Args:
            image: 캡처된 EMR 화면 이미지

        Returns:
            dict: 추출된 정보
        """
        # 이미지 전처리 (속도 및 정확도 향상)
        if self.enable_preprocessing:
            processed_image = self.preprocess_image(image)
        else:
            processed_image = image

        # OCR로 텍스트 추출 (속도 최적화)
        ocr_results = self.reader.readtext(
            processed_image,
            paragraph=False,
            batch_size=10,
            workers=0,
            decoder='greedy'
        )

        # 텍스트만 추출
        all_text = ' '.join([text for _, text, _ in ocr_results])

        # 정보 추출
        info = {
            'patient_id': self._extract_patient_id(all_text),
            'patient_name': self._extract_patient_name(ocr_results),
            'treatment_room': self._extract_treatment_room(all_text),
            'team': self._extract_team(all_text),
            'treatment_site': self._extract_treatment_site(all_text),
            'dose': self._extract_dose(all_text),
            'fraction': self._extract_fraction(all_text),
            'rt_method': self._extract_rt_method(all_text),
            'is_rf': self._check_rf(all_text),
            'image_guide': self._extract_image_guide(all_text),
            'gating': self._extract_gating(all_text),
            '_ocr_raw_text': all_text  # 디버그용 원본 OCR 텍스트
        }

        return info

    def _extract_patient_id(self, text):
        """환자 등록번호 추출 (8자리 숫자)"""
        # 71046541 형식
        match = re.search(r'\b(\d{8})\b', text)
        return match.group(1) if match else None

    def _extract_patient_name(self, ocr_results):
        """환자 이름 추출"""
        # 등록번호 바로 다음에 나오는 한글 텍스트
        # 또는 특정 위치의 한글 이름
        for i, (bbox, text, conf) in enumerate(ocr_results):
            # 8자리 숫자 다음에 나오는 한글
            if re.match(r'\d{8}', text):
                if i + 1 < len(ocr_results):
                    next_text = ocr_results[i + 1][1]
                    # 한글 이름 패턴
                    if re.match(r'^[가-힣]{2,4}$', next_text):
                        return next_text

        # 대체 방법: Overall 위쪽의 한글 이름 찾기
        for bbox, text, conf in ocr_results:
            if re.match(r'^[가-힣]{2,4}$', text) and conf > 0.5:
                return text

        return None

    def _extract_treatment_room(self, text):
        """치료실 번호 추출 (2TR -> 2치료실)"""
        match = re.search(r'(\d+)TR', text)
        if match:
            room_num = match.group(1)
            return f"{room_num}치료실"
        return None

    def _extract_team(self, text):
        """팀 정보 추출 (A~Z팀)"""
        # "Z팀", "A 팀", "2TR Z" 등의 패턴
        match = re.search(r'\b([A-Z])팀', text)
        if match:
            return f"{match.group(1)}팀"

        # "2TR 다음에 나오는 알파벳" 패턴
        match = re.search(r'\d+TR[^\w]*([A-Z])\b', text)
        if match:
            return f"{match.group(1)}팀"

        return None

    def _extract_treatment_site(self, text):
        """치료 부위 추출 (다양한 해부학적 부위 지원 - 100+ 부위)"""
        # 패턴 0: Rt/Lt/Both 포함 복합 부위 (가장 우선순위 높음)
        # "Rt thigh", "Lt Pleura", "Both lung", "Right femur" 등
        match = re.search(r'\b((?:Rt|Lt|Both|Right|Left)\s+[A-Za-z]+)\b', text, re.IGNORECASE)
        if match:
            site = match.group(1)
            # 첫 글자만 대문자로 변환
            parts = site.split()
            if len(parts) == 2:
                site = f"{parts[0].capitalize()}_{parts[1].lower()}"
            else:
                site = site.capitalize()
            if len(site) >= 4:  # 최소 4글자 이상 (Rt_x 형태)
                return site

        # 패턴 1: Overall 라인에서 추출
        # "Overall RLL:" 또는 "RLL: 39.6Gy" 패턴
        match = re.search(r'Overall[:\s]+([A-Za-z\s]+)[:\s]', text)
        if match:
            site = match.group(1).strip()
            if len(site) >= 2:  # 최소 2글자 이상
                return site.replace(' ', '_').capitalize()

        # 패턴 2: FINAL / RLL / 패턴
        match = re.search(r'FINAL[^\w]*/?[^\w]*([A-Za-z\s]+)[^\w]*/[^\w]*', text)
        if match:
            site = match.group(1).strip()
            if len(site) >= 2:
                return site.replace(' ', '_').capitalize()

        # 패턴 3: "부위:" 또는 "Site:" 다음의 텍스트
        match = re.search(r'(?:부위|Site|Location)[:\s]+([A-Za-z\s]+)\b', text, re.IGNORECASE)
        if match:
            return match.group(1).strip().replace(' ', '_').capitalize()

        # 패턴 4: 일반적인 치료 부위 (우선순위 순) - 100+ 부위 지원
        # 긴 단어부터 검색 (PROSTATE가 PROS로 잘못 인식되는 것 방지)
        common_sites = [
            # === 특수 대규모 조사 (Large Field Radiotherapy) ===
            r'\b(TBI)\b',    # Total Body Irradiation
            r'\b(TSEI)\b',   # Total Skin Electron Irradiation
            r'\b(UHBI)\b',   # Upper Hemibody Irradiation
            r'\b(LHBI)\b',   # Lower Hemibody Irradiation
            r'\b(HBI)\b',    # Hemibody Irradiation
            r'\b(TMI)\b',    # Total Marrow Irradiation
            r'\b(TMLI)\b',   # Total Marrow and Lymphoid Irradiation
            r'\b(TLI)\b',    # Total Lymphoid Irradiation
            r'\b(TNI)\b',    # Total Nodal Irradiation

            # === 두경부 (Head & Neck) - 세분화 ===
            r'\b(NASOPHARYNX)\b',
            r'\b(OROPHARYNX)\b',
            r'\b(HYPOPHARYNX)\b',
            r'\b(PARANASAL)\b',        # Paranasal sinuses
            r'\b(NASAL CAVITY)\b',     # Nasal cavity
            r'\b(NASAL VESTIBULE)\b',  # Nasal vestibule
            r'\b(ORAL CAVITY)\b',      # Oral cavity
            r'\b(SUPRAGLOTTIC)\b',     # Supraglottic larynx
            r'\b(VOCAL CORD)\b',       # Vocal cord
            r'\b(PAROTID)\b',          # Parotid glands
            r'\b(PARAGANGLIOMA)\b',    # Paragangliomas
            r'\b(LARYNX)\b',
            r'\b(TONGUE)\b',
            r'\b(THYROID)\b',

            # === 흉부 (Thorax) ===
            r'\b(MEDIASTINUM)\b',
            r'\b(ESOPHAGUS)\b',
            r'\b(PLEURA)\b',           # 흉막
            r'\b(PLEURAL CAVITY)\b',   # 흉막강
            r'\b(THORAX)\b',

            # === 폐 (Lung) ===
            r'\b(LUNG)\b',
            r'\b(RLL)\b',   # Right Lower Lobe
            r'\b(LLL)\b',   # Left Lower Lobe
            r'\b(RUL)\b',   # Right Upper Lobe
            r'\b(LUL)\b',   # Left Upper Lobe
            r'\b(RML)\b',   # Right Middle Lobe

            # === 복부 (Abdomen) ===
            r'\b(STOMACH)\b',
            r'\b(PANCREAS)\b',
            r'\b(LIVER)\b',
            r'\b(GALLBLADDER)\b',       # 담낭
            r'\b(SMALL BOWEL)\b',       # 소장
            r'\b(COLON)\b',             # 결장
            r'\b(EXTRAHEPATIC)\b',      # 간외담도
            r'\b(RETROPERITONEUM)\b',   # 후복막
            r'\b(PERITONEUM)\b',        # 복막
            r'\b(ABDOMEN)\b',
            r'\b(KIDNEY)\b',

            # === 골반 (Pelvis) ===
            r'\b(PROSTATE)\b',
            r'\b(RECTUM)\b',
            r'\b(CERVIX)\b',
            r'\b(UTERUS)\b',
            r'\b(BLADDER)\b',
            r'\b(OVARY)\b',    # 난소
            r'\b(TESTIS)\b',   # 고환
            r'\b(VAGINA)\b',   # 질
            r'\b(VULVA)\b',    # 외음부
            r'\b(PENIS)\b',    # 음경
            r'\b(PELVIS)\b',

            # === 유방 ===
            r'\b(BREAST)\b',

            # === 뇌/두개강 (Brain/CNS) ===
            r'\b(BRAIN)\b',
            r'\b(GBM)\b',              # Glioblastoma
            r'\b(PITUITARY)\b',        # 뇌하수체
            r'\b(ACOUSTIC)\b',         # 청신경종양
            r'\b(CNS)\b',              # Central Nervous System

            # === 척추 (Spine) ===
            r'\b(C-SPINE)\b',  # Cervical spine
            r'\b(T-SPINE)\b',  # Thoracic spine
            r'\b(L-SPINE)\b',  # Lumbar spine
            r'\b(CSPINE)\b',
            r'\b(TSPINE)\b',
            r'\b(LSPINE)\b',
            r'\b(SPINE)\b',

            # === 골격계 (Bones) ===
            r'\b(FEMUR)\b',     # 대퇴골
            r'\b(TIBIA)\b',     # 경골
            r'\b(FIBULA)\b',    # 비골
            r'\b(HUMERUS)\b',   # 상완골
            r'\b(RADIUS)\b',    # 요골
            r'\b(ULNA)\b',      # 척골
            r'\b(SCAPULA)\b',   # 견갑골
            r'\b(CLAVICLE)\b',  # 쇄골
            r'\b(STERNUM)\b',   # 흉골
            r'\b(RIBS?)\b',     # 늑골
            r'\b(ORBIT)\b',     # 안와
            r'\b(MAXILLA)\b',   # 상악골
            r'\b(MANDIBLE)\b',  # 하악골
            r'\b(SACRUM)\b',    # 천골
            r'\b(COCCYX)\b',    # 미골
            r'\b(ILIUM)\b',     # 장골
            r'\b(ISCHIUM)\b',   # 좌골
            r'\b(PUBIS)\b',     # 치골
            r'\b(BONE)\b',

            # === 사지 (Extremities) ===
            r'\b(THIGH)\b',         # 대퇴부
            r'\b(UPPER LIMB)\b',    # 상지
            r'\b(LOWER LIMB)\b',    # 하지
            r'\b(SOFT TISSUE)\b',   # 연부조직
            r'\b(SARCOMA)\b',       # 육종

            # === 눈/안과 ===
            r'\b(TAO)\b',    # Thyroid-Associated Ophthalmopathy
            r'\b(EYE)\b',    # 안구

            # === 피부 ===
            r'\b(SKIN)\b',

            # === 기타 부위 ===
            r'\b(HEAD)\b',
            r'\b(NECK)\b',
            r'\b(LYMPH NODE)\b',  # 림프절

            # === 약어 (Abbreviations) ===
            r'\b(HN)\b',    # Head and Neck
            r'\b(GI)\b',    # Gastrointestinal
            r'\b(GU)\b',    # Genitourinary
            r'\b(SRS)\b',   # Stereotactic Radiosurgery
            r'\b(SBRT)\b',  # SBRT는 RT method이지만 부위로 잘못 인식될 수 있음
        ]

        for pattern in common_sites:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                site = match.group(1)
                # SBRT는 RT method이므로 제외
                if site.upper() == 'SBRT':
                    continue
                # 공백을 언더스코어로 변환하고 첫 글자만 대문자
                return site.replace(' ', '_').capitalize()

        # 패턴 5: 한글 부위명 감지 후 영어로 매핑
        korean_to_english = {
            r'전립선': 'Prostate',
            r'유방': 'Breast',
            r'직장': 'Rectum',
            r'골반': 'Pelvis',
            r'폐': 'Lung',
            r'간': 'Liver',
            r'뇌': 'Brain',
            r'척추': 'Spine',
            r'식도': 'Esophagus',
            r'위': 'Stomach',
            r'췌장': 'Pancreas',
            r'자궁경부': 'Cervix',
            r'자궁': 'Uterus',
            r'방광': 'Bladder',
            r'신장': 'Kidney',
            r'두경부': 'Hn',
            r'갑상선': 'Thyroid',
        }

        for korean, english in korean_to_english.items():
            if re.search(korean, text):
                return english

        # 패턴 6: Gy 앞에 나오는 2글자 이상 영문 (마지막 수단)
        match = re.search(r'\b([A-Z]{2,})\b[^\w]*[\d.]+\s*Gy', text)
        if match:
            site = match.group(1)
            # 팀명(A, Z 등 단일 문자) 제외
            if len(site) >= 2:
                # 제외할 키워드 (RT method 등)
                exclude_keywords = ['SBRT', 'IMRT', 'VMAT', 'SRS', 'FINAL', 'PLAN']
                if site not in exclude_keywords:
                    return site.capitalize()

        return None

    def _extract_dose(self, text):
        """선량 추출 - 모든 가능한 형식 지원 (예: 39.6Gy, 3960cGy, 39.6 등)"""
        # 패턴 1: cGy 단위 (3960cGy -> 39.6Gy 변환)
        match = re.search(r'([\d.]+)\s*c[Gg][Yy]', text)
        if match:
            cgy_value = float(match.group(1))
            gy_value = cgy_value / 100
            return f"{gy_value}Gy"

        # 패턴 2: Gy 단위 (모든 대소문자 조합: Gy, gy, GY, gY)
        match = re.search(r'([\d.]+)\s*[Gg][Yy]', text)
        if match:
            return f"{match.group(1)}Gy"

        # 패턴 3: "Dose:", "Total:" 등의 라벨과 함께
        match = re.search(r'(?:Dose|Total|선량)[:\s]+([\d.]+)\s*[Gg][Yy]?', text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}Gy"

        # 패턴 4: 숫자만 있는 경우 (30-80 범위의 소수점 포함 숫자 - 일반적인 선량 범위)
        # Fraction 수와 혼동 방지를 위해 소수점이 있는 경우만
        match = re.search(r'\b((?:[3-7]\d|80)\.[\d]+)\b', text)
        if match:
            return f"{match.group(1)}Gy"

        return None

    def _extract_fraction(self, text):
        """Fraction 수 추출 - 모든 가능한 형식 지원"""
        # 패턴 1: fx/Fx/FX/fX (모든 대소문자 조합)
        match = re.search(r'(\d+)\s*[fF][xX]', text)
        if match:
            return f"{match.group(1)}Fx"

        # 패턴 2: fr, frac, fraction
        match = re.search(r'(\d+)\s*(?:[fF]rac?(?:tion)?s?)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}Fx"

        # 패턴 3: "Fraction:", "#", "회" 등의 라벨과 함께
        match = re.search(r'(?:Fraction|#|횟수|회)[:\s]*(\d+)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}Fx"

        # 패턴 4: "in 18 fractions" 또는 "/ 18" 형식
        match = re.search(r'(?:in|/)\s*(\d+)\s*(?:fraction|fx|fr)?', text, re.IGNORECASE)
        if match:
            fx_num = int(match.group(1))
            # 일반적인 fraction 범위 (1-40 정도)
            if 1 <= fx_num <= 40:
                return f"{fx_num}Fx"

        return None

    def _extract_rt_method(self, text):
        """RT Method 추출 (SBRT만 표시)"""
        # SBRT인 경우에만 반환
        if re.search(r'\bSBRT\b', text, re.IGNORECASE):
            return 'SBRT'
        return None

    def _check_rf(self, text):
        """RF 여부 확인"""
        # FINAL-RF 패턴 찾기
        if re.search(r'FINAL[- ]RF', text, re.IGNORECASE):
            return True
        return False

    def _extract_image_guide(self, text):
        """Image Guide 코드 추출 (G1->G2 등)"""
        # G1->G2, G1 -> G2 등의 패턴
        match = re.search(r'(G\d+)\s*-+>\s*(G\d+)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}->{match.group(2)}"

        # 단일 G 코드
        match = re.search(r'\b(G\d+)\b', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_gating(self, text):
        """Gating 방법 추출"""
        # No gating with monitoring
        if re.search(r'No\s+gating\s+with\s+monitoring', text, re.IGNORECASE):
            return 'NGM'

        # No Gating
        if re.search(r'No\s+[Gg]ating', text, re.IGNORECASE):
            return 'NG'

        # 다른 gating 방법이 있다면 추가
        return None

    def validate_info(self, info):
        """추출된 정보 검증"""
        required_fields = ['patient_id', 'patient_name', 'treatment_site']

        for field in required_fields:
            if not info.get(field):
                return False, f"필수 정보 누락: {field}"

        return True, "OK"
