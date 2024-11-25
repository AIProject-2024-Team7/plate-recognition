# YOLO Detection 코드


import cv2
from ultralytics import YOLO

class LicensePlateDetector:
    #YOLO 모델을 이용한 번호판 감지
    def __init__(self, model_path="yolo_weights.pt", input_size=(640, 640)):
        self.model = YOLO(model_path)  # YOLO 모델 로드
        self.model.eval()  # 평가 모드
        self.input_size = input_size  # 입력 이미지 크기

    def preprocess_image(self, image):
        """
        입력 이미지를 YOLO 모델에 맞게 크기 조정.
        :param image: 원본 이미지
        :return: 크기 조정된 이미지
        """
        resized_image = cv2.resize(image, self.input_size)  # (640, 640)로 고정
        return resized_image

    def detect(self, image):
        """
        번호판 영역을 감지하고 크롭된 이미지와 좌표를 반환.
        :param image: 입력 이미지
        :return: 번호판 이미지 리스트와 Bounding Box 좌표
        """
        image = self.preprocess_image(image)
        results = self.model(image)
        cropped_images = []
        bounding_boxes = []

        for box in results[0].boxes.xyxy:  # 감지된 Bounding Box 좌표
            x1, y1, x2, y2 = map(int, box)  # 좌표 정수화
            cropped_images.append(image[y1:y2, x1:x2])  # 번호판 크롭
            bounding_boxes.append((x1, y1, x2, y2))

        return cropped_images, bounding_boxes
