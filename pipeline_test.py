import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams 
from matplotlib import font_manager
from ultralytics import YOLO

rcParams['font.family'] = 'NanumGothic'

cls_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '가', '나', '다', '라', '마', '거', '너', '더', '러', '머',
        '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
        '소', '오', '조', '구', '누', '두', '루', '무', '부', '수',
        '우', '주', '바', '사', '아', '자', '허', '하', '호', '배']

def img_crop(img, bbox):
    print(bbox)
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2],:]


class Pipeline(nn.Module):

    def __init__(self):

        super().__init__()

        self.detection_model = YOLO('./license_plate_noisy/yolov8_finetuned/weights/best.pt').to('cuda:0')
        self.recognition_model = YOLO('./recognition/yolov8_finetuned_recognition/weights/best.pt').to('cuda:0')

        self.detection_model.eval()
        self.recognition_model.eval()

    def forward(self, img):

        img_h,  img_w, _ = img.shape

        results = self.detection_model(img)[0].boxes.xyxy
        
        final_result = ''
        for res in results:
            bbox = res.to(torch.int).to('cpu').numpy().squeeze()
            cropped = img_crop(img, bbox)

            results = self.recognition_model(cropped)[0].boxes
            sort_target = results.xyxy[:,0]
            sorted_indices = torch.argsort(sort_target)

            pred = results.cls.to(torch.int)[sorted_indices].to('cpu').numpy()
            print([cls_mapping[i] for i in pred])
            final_result = final_result.join(cls_mapping[i] for i in pred)
            final_result += ', '

        return final_result


prototype = Pipeline()

image_dir = './datasets/detection/test/images'

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]



for i, img_file in enumerate(image_files):

    img_path = os.path.join(image_dir, img_file)

    img = cv2.imread(img_path)
    img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.clf()
    plt.imshow(img_plt)

    pred = prototype(img)

    plt.title(pred)
    plt.savefig(f'test_{i}.png')