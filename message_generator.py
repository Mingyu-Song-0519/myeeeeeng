"""
메시지 생성 모듈
"""


class MessageGenerator:
    def __init__(self):
        """메시지 생성기 초기화"""
        pass

    def generate_message(self, patient_info):
        """환자 정보로부터 메시지 생성

        Args:
            patient_info: OCR로 추출한 환자 정보 딕셔너리

        Returns:
            str: 생성된 메시지
        """
        # 기본 인사말
        greeting = "안녕하세요 선생님~"

        # 메시지 본문 구성
        parts = []

        # 치료실
        if patient_info.get('treatment_room'):
            parts.append(patient_info['treatment_room'])

        # 팀
        if patient_info.get('team'):
            parts.append(patient_info['team'])

        # 환자 이름
        if patient_info.get('patient_name'):
            parts.append(f"{patient_info['patient_name']}님")

        # 등록번호
        if patient_info.get('patient_id'):
            parts.append(patient_info['patient_id'])

        # 치료 부위
        if patient_info.get('treatment_site'):
            parts.append(patient_info['treatment_site'])

        # RT Method (SBRT만 표시)
        if patient_info.get('rt_method') == 'SBRT':
            parts.append('SBRT')

        # RF 여부
        if patient_info.get('is_rf'):
            parts.append('RF')

        # 선량/Fx (항상 39.6Gy/18Fx 형식으로 통일)
        dose = patient_info.get('dose')
        fraction = patient_info.get('fraction')

        if dose and fraction:
            parts.append(f"{dose}/{fraction}")
        elif dose:
            parts.append(dose)
        elif fraction:
            parts.append(fraction)

        # Image Guide
        if patient_info.get('image_guide'):
            parts.append(patient_info['image_guide'])

        # Gating
        if patient_info.get('gating'):
            parts.append(patient_info['gating'])

        # 메시지 조합
        main_content = ' '.join(parts)

        # 마무리 멘트 (SBRT 환자는 다른 표현)
        if patient_info.get('rt_method') == 'SBRT':
            closing = "첫 치료 영상 저장했습니다.\n혹시 지금 오실 수 있으신가요?"
        else:
            closing = "첫 치료 영상 저장했습니다.\nOffline review 확인 부탁드립니다~"

        # 전체 메시지
        message = f"{greeting}\n{main_content}\n{closing}"

        return message

    def generate_debug_message(self, patient_info):
        """디버깅용 상세 메시지 생성"""
        lines = ["=== 추출된 환자 정보 ==="]

        for key, value in patient_info.items():
            lines.append(f"{key}: {value}")

        lines.append("\n=== 생성된 메시지 ===")
        lines.append(self.generate_message(patient_info))

        return '\n'.join(lines)
