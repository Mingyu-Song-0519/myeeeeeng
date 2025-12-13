"""
화면 캡처 및 템플릿 매칭 모듈
"""

import mss
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ScreenCapture:
    def __init__(self, template_dir="images", max_width=1920, max_height=1080, roi=None):
        """화면 캡처 초기화

        Args:
            template_dir: 템플릿 이미지 디렉토리
            max_width: 최대 너비 (해상도 제한, 속도 향상)
            max_height: 최대 높이 (해상도 제한, 속도 향상)
            roi: Region of Interest {'x': int, 'y': int, 'width': int, 'height': int}
        """
        self.template_dir = Path(template_dir)
        # mss 객체를 초기화 시 생성하지 않고 캡처 시마다 생성 (스레드 안전성)
        self.max_width = max_width
        self.max_height = max_height
        self.roi = roi  # ROI 설정

    def capture_screen(self, monitor_number=1, resize=True):
        """화면 캡처

        Args:
            monitor_number: 모니터 번호 (1부터 시작)
            resize: 해상도 제한 적용 여부 (True: 속도 향상, False: 원본 해상도)

        Returns:
            numpy array: 캡처된 이미지
        """
        # ROI가 설정되어 있으면 ROI 영역만 캡처
        if self.roi:
            return self.capture_region(
                self.roi['x'],
                self.roi['y'],
                self.roi['width'],
                self.roi['height']
            )

        # 매번 새로운 mss 객체 생성 (스레드 안전성 보장)
        with mss.mss() as sct:
            monitor = sct.monitors[monitor_number]
            screenshot = sct.grab(monitor)

            # BGRA to BGR 변환
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 해상도 제한 (OCR 속도 향상)
        if resize:
            height, width = img.shape[:2]

            # 최대 해상도를 초과하는 경우 리사이즈
            if width > self.max_width or height > self.max_height:
                # 종횡비 유지하면서 리사이즈
                scale = min(self.max_width / width, self.max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)

                img = cv2.resize(img, (new_width, new_height),
                               interpolation=cv2.INTER_AREA)  # INTER_AREA: 축소 시 최적

        return img

    def set_roi(self, roi):
        """ROI 설정

        Args:
            roi: {'x': int, 'y': int, 'width': int, 'height': int} or None
        """
        self.roi = roi
        if roi:
            print(f"ROI 설정됨: {roi}")
        else:
            print("ROI 해제됨 (전체 화면 캡처)")

    def get_roi(self):
        """현재 ROI 가져오기

        Returns:
            dict: ROI 정보 or None
        """
        return self.roi

    def find_template_in_screen(self, template_path, threshold=0.7):
        """화면에서 템플릿 이미지 찾기

        Args:
            template_path: 템플릿 이미지 경로
            threshold: 매칭 임계값 (0~1)

        Returns:
            tuple: (찾았는지 여부, 좌표 (x, y, w, h))
        """
        # 화면 캡처
        screen = self.capture_screen()

        # 템플릿 로드
        template = cv2.imread(str(template_path))
        if template is None:
            print(f"템플릿을 로드할 수 없습니다: {template_path}")
            return False, None

        # 템플릿 매칭
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 임계값 이상이면 찾음
        if max_val >= threshold:
            h, w = template.shape[:2]
            x, y = max_loc
            return True, (x, y, w, h)

        return False, None

    def is_emr_window_active(self, threshold=0.6):
        """EMR 창이 활성화되어 있는지 확인

        Args:
            threshold: 매칭 임계값

        Returns:
            bool: EMR 창 활성 여부
        """
        # images 폴더의 모든 이미지로 확인
        for template_file in self.template_dir.glob("*.jpg"):
            found, _ = self.find_template_in_screen(template_file, threshold)
            if found:
                return True

        for template_file in self.template_dir.glob("*.png"):
            found, _ = self.find_template_in_screen(template_file, threshold)
            if found:
                return True

        return False

    def capture_region(self, x, y, width, height):
        """특정 영역 캡처

        Args:
            x, y: 시작 좌표
            width, height: 크기

        Returns:
            numpy array: 캡처된 이미지
        """
        monitor = {"top": y, "left": x, "width": width, "height": height}

        # 매번 새로운 mss 객체 생성 (스레드 안전성 보장)
        with mss.mss() as sct:
            screenshot = sct.grab(monitor)

            # BGRA to BGR 변환
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    def save_screenshot(self, filename="screenshot.png"):
        """스크린샷 저장 (디버깅용)"""
        img = self.capture_screen()
        cv2.imwrite(filename, img)
        print(f"스크린샷 저장: {filename}")
