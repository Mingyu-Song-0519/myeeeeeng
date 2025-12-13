"""
간단한 테스트 프로그램 - 빌드가 제대로 되는지 확인
"""

import sys
import time

print("=" * 60)
print("EMR Helper - Simple Test")
print("=" * 60)
print()
print("If you can see this message, the build works!")
print("Python executable:", sys.executable)
print()

# 기본 모듈 테스트
try:
    import pyautogui
    print("✓ pyautogui imported")
except Exception as e:
    print("✗ pyautogui failed:", e)

try:
    import keyboard
    print("✓ keyboard imported")
except Exception as e:
    print("✗ keyboard failed:", e)

try:
    import cv2
    print("✓ opencv imported")
except Exception as e:
    print("✗ opencv failed:", e)

try:
    import easyocr
    print("✓ easyocr imported")
except Exception as e:
    print("✗ easyocr failed:", e)

print()
print("=" * 60)
print("Test completed!")
print("=" * 60)
print()
print("Press ESC to exit...")

import keyboard
keyboard.wait('esc')
