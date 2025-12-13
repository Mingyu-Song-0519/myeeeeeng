"""
EMR 자동 텍스트 입력 도우미
화면을 OCR로 인식하여 자동으로 메시지를 생성하고 입력하는 프로그램
"""

import pyautogui
import keyboard
import json
import time
import sys
import os
from pathlib import Path

from screen_capture import ScreenCapture
from ocr_extractor import EMRDataExtractor
from message_generator import MessageGenerator
from roi_selector import ROISelector, save_roi_to_config, load_roi_from_config


def get_resource_path(relative_path):
    """PyInstaller로 빌드된 실행 파일에서 리소스 경로 가져오기

    Args:
        relative_path: 상대 경로

    Returns:
        str: 절대 경로
    """
    try:
        # PyInstaller가 생성한 임시 폴더
        base_path = sys._MEIPASS
    except Exception:
        # 개발 환경
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class EMRHelper:
    def __init__(self, config_path="config.json"):
        """EMR 도우미 초기화"""
        self.config = self.load_config(config_path)
        self.running = False

        # OCR 모드 설정
        self.ocr_mode = self.config.get('ocr_mode', True)

        # 캐싱 설정
        self.enable_cache = self.config.get('enable_cache', True)
        self.cache = {}  # {patient_id: (patient_info, timestamp)}
        self.cache_timeout = self.config.get('cache_timeout_seconds', 300)  # 5분

        if self.ocr_mode:
            print("OCR 모듈 로딩 중... (최초 실행 시 시간이 걸릴 수 있습니다)")
            # PyInstaller 환경에서 올바른 이미지 경로 찾기
            images_path = get_resource_path("images")

            # ROI 로드
            roi = self.config.get('roi')
            if roi:
                print(f"ROI 설정 로드됨: {roi}")

            self.screen_capture = ScreenCapture(template_dir=images_path, roi=roi)
            self.ocr_extractor = EMRDataExtractor()
            self.message_generator = MessageGenerator()
            print("OCR 모듈 로딩 완료!")

    def load_config(self, config_path):
        """설정 파일 로드"""
        # PyInstaller 환경에서 올바른 경로 찾기
        config_path = get_resource_path(config_path)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {config_path}")
            print(f"현재 작업 디렉토리: {os.getcwd()}")
            print(f"리소스 경로: {getattr(sys, '_MEIPASS', '개발 환경')}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"설정 파일 형식이 잘못되었습니다: {config_path}")
            sys.exit(1)

    def get_active_window_title(self):
        """현재 활성 윈도우 제목 가져오기 (Windows)"""
        try:
            import win32gui
            window = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(window)
        except ImportError:
            print("경고: pywin32가 설치되지 않았습니다. 윈도우 감지 기능이 제한됩니다.")
            return ""

    def is_emr_window_active(self):
        """EMR 창이 활성화되어 있는지 확인

        Returns:
            bool: 방사선치료[방사선종양] 창이 활성화되어 있으면 True
        """
        window_title = self.get_active_window_title()

        # 정확한 패턴 매칭: "방사선치료[방사선종양]" 포함 여부 확인
        if "방사선치료" in window_title and "방사선종양" in window_title:
            return True

        # 추가 패턴: 환자 정보 형식 확인 (선택사항)
        # 예: "(12345678 홍길동 남/45) 방사선치료[방사선종양]"
        import re
        # 8자리 숫자 + 이름 + 방사선치료[방사선종양] 패턴
        pattern = r'\(\d{8}\s+[가-힣]+\s+[남녀]/\d+\).*방사선치료\[방사선종양\]'
        if re.search(pattern, window_title):
            return True

        return False

    def get_window_info(self):
        """현재 활성 창 정보 가져오기 (디버깅용)"""
        window_title = self.get_active_window_title()
        is_valid = self.is_emr_window_active()
        return window_title, is_valid

    def get_cached_info(self, patient_id):
        """캐시에서 환자 정보 가져오기"""
        if not self.enable_cache or patient_id not in self.cache:
            return None

        cached_info, cached_time = self.cache[patient_id]

        # 캐시 만료 확인
        if time.time() - cached_time > self.cache_timeout:
            del self.cache[patient_id]
            return None

        return cached_info

    def set_cached_info(self, patient_id, patient_info):
        """캐시에 환자 정보 저장"""
        if self.enable_cache and patient_id:
            self.cache[patient_id] = (patient_info, time.time())

    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        print("캐시가 초기화되었습니다.")

    def type_message_korean(self, message):
        """한글 메시지 입력 (클립보드 사용)"""
        import pyperclip

        click_before_type = self.config['auto_input']['click_before_type']

        # 클립보드에 복사
        pyperclip.copy(message)

        # 클릭 후 입력 옵션
        if click_before_type:
            time.sleep(0.5)
            pyautogui.click()
            time.sleep(0.2)

        # Ctrl+V로 붙여넣기
        pyautogui.hotkey('ctrl', 'v')
        print(f"메시지 입력 완료: {len(message)}자")

    def extract_and_generate_message(self, use_cache=True):
        """화면에서 정보를 추출하고 메시지 생성

        Args:
            use_cache: 캐시 사용 여부
        """
        try:
            print("\n화면 캡처 중...")
            # 화면 캡처
            screen = self.screen_capture.capture_screen()

            # 빠른 환자 ID 추출 (캐시 확인용)
            patient_id = None
            if use_cache and self.enable_cache:
                # 간단한 OCR로 환자 ID만 먼저 추출
                import re
                # 화면에서 환자 ID 영역만 빠르게 스캔
                quick_results = self.ocr_extractor.reader.readtext(
                    screen, paragraph=False, workers=0, decoder='greedy'
                )
                quick_text = ' '.join([text for _, text, _ in quick_results])
                match = re.search(r'\b(\d{8})\b', quick_text)
                if match:
                    patient_id = match.group(1)

                    # 캐시에서 확인
                    cached_info = self.get_cached_info(patient_id)
                    if cached_info:
                        print(f"캐시에서 환자 정보 로드: {patient_id} (OCR 건너뜀)")
                        patient_info = cached_info

                        # 메시지 생성
                        message = self.message_generator.generate_message(patient_info)

                        print("\n[캐시] 추출된 정보:")
                        print(f"  환자: {patient_info.get('patient_name')}({patient_info.get('patient_id')})")
                        print(f"  치료실: {patient_info.get('treatment_room')}, 팀: {patient_info.get('team')}")
                        print(f"  치료부위: {patient_info.get('treatment_site')}")

                        print("\n생성된 메시지:")
                        print("-" * 60)
                        print(message)
                        print("-" * 60)

                        return message

            print("OCR 처리 중... (시간이 걸릴 수 있습니다)")
            # OCR로 정보 추출
            patient_info = self.ocr_extractor.extract_patient_info(screen)

            # 정보 검증
            is_valid, msg = self.ocr_extractor.validate_info(patient_info)

            if not is_valid:
                print(f"오류: {msg}")
                print("추출된 정보:")
                for key, value in patient_info.items():
                    print(f"  {key}: {value}")
                return None

            # 캐시에 저장
            if patient_info.get('patient_id'):
                self.set_cached_info(patient_info['patient_id'], patient_info)

            # 메시지 생성
            message = self.message_generator.generate_message(patient_info)

            # 디버그 정보 출력
            if self.config.get('debug_mode', False):
                debug_msg = self.message_generator.generate_debug_message(patient_info)
                print(debug_msg)
            else:
                print("\n추출된 정보:")
                print(f"  환자: {patient_info.get('patient_name')}({patient_info.get('patient_id')})")
                print(f"  치료실: {patient_info.get('treatment_room')}, 팀: {patient_info.get('team')}")
                print(f"  치료부위: {patient_info.get('treatment_site')}")

            print("\n생성된 메시지:")
            print("-" * 60)
            print(message)
            print("-" * 60)

            return message

        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

    def on_hotkey_ocr(self):
        """OCR 모드 단축키 핸들러"""
        print(f"\n단축키 감지: {self.config['hotkey']}")

        # EMR 창 확인 (skip_window_check 설정에 따라)
        skip_check = self.config.get('skip_window_check', False)

        if not skip_check:
            window_title, is_valid = self.get_window_info()

            if not is_valid:
                print("=" * 60)
                print("⚠️  경고: 올바른 EMR 창이 아닙니다!")
                print("=" * 60)
                print(f"현재 활성 창: {window_title}")
                print("")
                print("✅ 올바른 창 형식:")
                print("   (환자번호 이름 성별/나이) 방사선치료[방사선종양]")
                print("")
                print("예시:")
                print("   (12345678 홍길동 남/45) 방사선치료[방사선종양]")
                print("=" * 60)
                print("\n💡 팁: ROI를 사용하는 경우 config.json에서")
                print("   \"skip_window_check\": true 로 설정하면")
                print("   창 검증을 건너뛸 수 있습니다.")
                print("\n프로그램은 계속 실행 중입니다. ESC를 눌러 종료하세요.")
                print("=" * 60)
                return

            print(f"✅ EMR 창 확인: {window_title[:50]}...")
        else:
            print("⚠️  창 검증 건너뜀 (skip_window_check=true)")

        # 메시지 생성
        message = self.extract_and_generate_message()

        if message:
            # 자동 입력 여부 확인
            if self.config.get('auto_type_after_extraction', True):
                print("\n메시지 입력 중...")
                self.type_message_korean(message)
                print("\n✅ 완료! 프로그램은 계속 실행 중입니다.")
            else:
                print("\n메시지가 클립보드에 복사되었습니다.")
                print("✅ 완료! 프로그램은 계속 실행 중입니다.")
                import pyperclip
                pyperclip.copy(message)
        else:
            print("메시지 생성에 실패했습니다.")
            print("프로그램은 계속 실행 중입니다. 다시 시도하거나 ESC를 눌러 종료하세요.")

    def on_hotkey_simple(self):
        """단순 모드 단축키 핸들러"""
        print(f"\n단축키 감지: {self.config['hotkey']}")

        # EMR 창 확인
        if not self.is_emr_window_active():
            print("경고: EMR 창이 활성화되지 않았습니다.")

        try:
            message = self.config['message_template']
            self.type_message_korean(message)
        except Exception as e:
            print(f"오류 발생: {e}")

    def on_hotkey_set_roi(self):
        """ROI 설정 단축키 핸들러 (Ctrl+Shift+R)"""
        print("\n=== ROI 설정 모드 ===")
        print("마우스로 드래그하여 ROI 영역을 선택하세요...")
        print("(ESC: 취소)")

        try:
            # ROI 선택 GUI 실행
            selector = ROISelector()
            roi = selector.select()

            if roi:
                # ROI 설정
                if self.ocr_mode and hasattr(self, 'screen_capture'):
                    self.screen_capture.set_roi(roi)

                # config.json에 저장
                save_roi_to_config(roi, "config.json")

                # config에도 반영
                self.config['roi'] = roi

                print("\n✅ ROI가 설정되었습니다!")
                print(f"   위치: ({roi['x']}, {roi['y']})")
                print(f"   크기: {roi['width']} x {roi['height']}")
                print("\n이제 Ctrl+Shift+A를 누르면 설정된 영역만 캡처합니다.")
                print("프로그램은 계속 실행 중입니다.")
            else:
                print("ROI 설정이 취소되었습니다.")
                print("프로그램은 계속 실행 중입니다.")

        except Exception as e:
            print(f"ROI 설정 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    def on_hotkey_clear_roi(self):
        """ROI 초기화 단축키 핸들러 (Ctrl+Shift+D)"""
        print("\n=== ROI 초기화 ===")

        try:
            # ROI 해제
            if self.ocr_mode and hasattr(self, 'screen_capture'):
                self.screen_capture.set_roi(None)

            # config.json에서 제거
            if 'roi' in self.config:
                del self.config['roi']

            config_path = get_resource_path("config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            print("✅ ROI가 초기화되었습니다.")
            print("이제 전체 화면을 캡처합니다.")
            print("프로그램은 계속 실행 중입니다.")

        except Exception as e:
            print(f"ROI 초기화 중 오류 발생: {e}")
            print("프로그램은 계속 실행 중입니다.")

    def run(self):
        """프로그램 실행"""
        print("=" * 60)
        print("EMR 자동 텍스트 입력 도우미")
        print("=" * 60)
        print(f"모드: {'OCR 자동 추출' if self.ocr_mode else '단순 입력'}")
        print(f"단축키: {self.config['hotkey']}")
        print("종료: ESC 키")
        print("-" * 60)

        if self.ocr_mode:
            print("화면에서 환자 정보를 자동으로 추출하여 메시지를 생성합니다.")
            print("")

            # 창 검증 설정 표시
            skip_check = self.config.get('skip_window_check', False)
            if not skip_check:
                print("⚠️  창 검증: 활성화")
                print("   (환자번호 이름 성별/나이) 방사선치료[방사선종양]")
                print("   다른 창에서 단축키를 누르면 경고 메시지가 표시됩니다.")
            else:
                print("✅ 창 검증: 비활성화 (모든 창에서 작동)")
            print("")

            # ROI 상태 표시
            roi = self.config.get('roi')
            if roi:
                print("📍 ROI 설정됨:")
                print(f"   위치: ({roi['x']}, {roi['y']})")
                print(f"   크기: {roi['width']} x {roi['height']}")
                print("   ➜ 설정된 영역만 캡처합니다")
            else:
                print("📍 ROI 미설정: 전체 화면 캡처")

            print("")
            print("추가 단축키:")
            print("   Ctrl+Shift+R: ROI 영역 설정")
            print("   Ctrl+Shift+D: ROI 초기화 (전체 화면)")
        else:
            print("메시지 미리보기:")
            print(self.config['message_template'])

        print("=" * 60)

        # 단축키 등록
        if self.ocr_mode:
            keyboard.add_hotkey(self.config['hotkey'], self.on_hotkey_ocr, suppress=False)
            keyboard.add_hotkey('ctrl+shift+r', self.on_hotkey_set_roi, suppress=False)
            keyboard.add_hotkey('ctrl+shift+d', self.on_hotkey_clear_roi, suppress=False)
        else:
            keyboard.add_hotkey(self.config['hotkey'], self.on_hotkey_simple, suppress=False)

        # ESC 키로 종료
        print("\n프로그램이 실행 중입니다... (ESC: 종료)")
        keyboard.wait('esc')

        print("\n프로그램을 종료합니다.")


def main():
    """메인 함수"""
    try:
        helper = EMRHelper()
        helper.run()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
