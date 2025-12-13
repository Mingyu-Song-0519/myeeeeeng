"""
OCR 테스트 스크립트
이미지 파일로 OCR 기능을 테스트합니다
"""

import cv2
import sys
from pathlib import Path
from ocr_extractor import EMRDataExtractor
from message_generator import MessageGenerator


def test_ocr_on_image(image_path):
    """이미지 파일로 OCR 테스트"""
    print("=" * 60)
    print(f"OCR 테스트: {image_path}")
    print("=" * 60)

    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return

    print("\nOCR 모듈 초기화 중...")
    extractor = EMRDataExtractor()
    generator = MessageGenerator()

    print("OCR 처리 중... (시간이 걸릴 수 있습니다)")
    patient_info = extractor.extract_patient_info(image)

    print("\n" + "=" * 60)
    print("추출된 정보:")
    print("=" * 60)
    for key, value in patient_info.items():
        print(f"{key:25} : {value}")

    print("\n" + "=" * 60)
    print("생성된 메시지:")
    print("=" * 60)
    message = generator.generate_message(patient_info)
    print(message)
    print("=" * 60)

    # 검증
    is_valid, msg = extractor.validate_info(patient_info)
    print(f"\n검증 결과: {msg}")


def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        # 명령행 인자로 이미지 경로 지정
        image_path = sys.argv[1]
    else:
        # 기본값: images 폴더의 세 번째 이미지
        image_path = "images/KakaoTalk_20251211_124226257_02.jpg"

    test_ocr_on_image(image_path)


if __name__ == "__main__":
    main()
