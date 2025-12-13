"""
ROI(Region of Interest) 선택 모듈
마우스 드래그로 화면에서 관심 영역을 선택
"""

import tkinter as tk
from PIL import ImageGrab, ImageTk
import json
import sys
import os


def get_resource_path(relative_path):
    """PyInstaller 환경에서 리소스 경로 가져오기"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class ROISelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)  # 반투명
        self.root.attributes('-topmost', True)

        # 화면 캡처
        self.screenshot = ImageGrab.grab()
        self.photo = ImageTk.PhotoImage(self.screenshot)

        # 캔버스 생성
        self.canvas = tk.Canvas(self.root, cursor="cross", bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 스크린샷 배경으로 표시
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # 선택 영역 변수
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selected_roi = None

        # 이벤트 바인딩
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Escape>", self.on_cancel)

        # 안내 텍스트
        info_text = "마우스로 드래그하여 ROI 영역을 선택하세요 (ESC: 취소)"
        self.canvas.create_text(
            self.screenshot.width // 2, 30,
            text=info_text,
            font=("Arial", 16, "bold"),
            fill="yellow",
            tags="info"
        )

    def on_press(self, event):
        """마우스 버튼 누름"""
        self.start_x = event.x
        self.start_y = event.y

        # 기존 사각형 제거
        if self.rect:
            self.canvas.delete(self.rect)

    def on_drag(self, event):
        """마우스 드래그"""
        if self.start_x is None or self.start_y is None:
            return

        # 기존 사각형 제거
        if self.rect:
            self.canvas.delete(self.rect)

        # 새 사각형 그리기
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            event.x, event.y,
            outline="red",
            width=3
        )

        # 크기 정보 표시
        width = abs(event.x - self.start_x)
        height = abs(event.y - self.start_y)
        size_text = f"크기: {width} x {height}"

        # 기존 크기 텍스트 제거
        self.canvas.delete("size")

        # 새 크기 텍스트
        self.canvas.create_text(
            (self.start_x + event.x) // 2,
            (self.start_y + event.y) // 2,
            text=size_text,
            font=("Arial", 14, "bold"),
            fill="yellow",
            tags="size"
        )

    def on_release(self, event):
        """마우스 버튼 놓음 - ROI 선택 완료"""
        if self.start_x is None or self.start_y is None:
            return

        # 좌표 정규화 (왼쪽 위가 시작점이 되도록)
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)

        width = x2 - x1
        height = y2 - y1

        # 최소 크기 검증 (너무 작은 영역 방지)
        if width < 50 or height < 50:
            print("ROI 영역이 너무 작습니다. (최소 50x50)")
            self.start_x = None
            self.start_y = None
            if self.rect:
                self.canvas.delete(self.rect)
            return

        self.selected_roi = {
            "x": x1,
            "y": y1,
            "width": width,
            "height": height
        }

        print(f"ROI 선택 완료: {self.selected_roi}")
        self.root.quit()

    def on_cancel(self, event):
        """ESC 키 - 취소"""
        print("ROI 선택이 취소되었습니다.")
        self.selected_roi = None
        self.root.quit()

    def select(self):
        """ROI 선택 시작

        Returns:
            dict: {'x': int, 'y': int, 'width': int, 'height': int} or None
        """
        self.root.mainloop()
        self.root.destroy()
        return self.selected_roi


def save_roi_to_config(roi, config_path="config.json"):
    """ROI를 config.json에 저장

    Args:
        roi: {'x': int, 'y': int, 'width': int, 'height': int}
        config_path: 설정 파일 경로
    """
    config_path = get_resource_path(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}

    config['roi'] = roi

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"ROI 설정이 저장되었습니다: {roi}")


def load_roi_from_config(config_path="config.json"):
    """config.json에서 ROI 로드

    Args:
        config_path: 설정 파일 경로

    Returns:
        dict: {'x': int, 'y': int, 'width': int, 'height': int} or None
    """
    config_path = get_resource_path(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('roi')
    except (FileNotFoundError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    # 테스트
    selector = ROISelector()
    roi = selector.select()

    if roi:
        print(f"선택된 ROI: {roi}")
        save_roi_to_config(roi)
    else:
        print("ROI가 선택되지 않았습니다.")
