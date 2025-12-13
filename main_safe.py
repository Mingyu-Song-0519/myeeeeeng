"""
EMR 자동 텍스트 입력 도우미 - 안전 버전
화면을 OCR로 인식하여 자동으로 메시지를 생성하고 입력하는 프로그램
에러 로깅 포함
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# 에러 로깅 설정
def setup_logging():
    """에러를 파일로 기록"""
    log_file = Path("error.log")

    class Logger:
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_file, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger()
    sys.stderr = Logger()

    print(f"\n{'='*60}")
    print(f"EMR Helper - Started at {datetime.now()}")
    print(f"{'='*60}\n")

def main():
    """메인 함수"""
    try:
        setup_logging()

        print("Importing modules...")

        # 필수 모듈 import
        try:
            import pyautogui
            print("✓ pyautogui")
        except ImportError as e:
            print(f"✗ pyautogui failed: {e}")
            raise

        try:
            import keyboard
            print("✓ keyboard")
        except ImportError as e:
            print(f"✗ keyboard failed: {e}")
            raise

        try:
            import json
            print("✓ json")
        except ImportError as e:
            print(f"✗ json failed: {e}")
            raise

        try:
            import time
            print("✓ time")
        except ImportError as e:
            print(f"✗ time failed: {e}")
            raise

        try:
            from screen_capture import ScreenCapture
            print("✓ screen_capture")
        except ImportError as e:
            print(f"✗ screen_capture failed: {e}")
            print("Make sure screen_capture.py is in the same directory")
            raise

        try:
            from ocr_extractor import EMRDataExtractor
            print("✓ ocr_extractor")
        except ImportError as e:
            print(f"✗ ocr_extractor failed: {e}")
            print("Make sure ocr_extractor.py is in the same directory")
            raise

        try:
            from message_generator import MessageGenerator
            print("✓ message_generator")
        except ImportError as e:
            print(f"✗ message_generator failed: {e}")
            print("Make sure message_generator.py is in the same directory")
            raise

        print("\nAll modules imported successfully!")
        print()

        # 실제 프로그램 실행
        from main import EMRHelper
        print("Starting EMR Helper...")
        helper = EMRHelper()
        helper.run()

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR OCCURRED!")
        print(f"{'='*60}\n")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"\n{'='*60}")
        print("Error has been logged to error.log")
        print(f"{'='*60}\n")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
