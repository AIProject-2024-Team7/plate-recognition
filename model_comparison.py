from ultralytics import YOLO
import yaml

model1_path = './license_plate_old_dataset/yolov8_finetuned_detection/weights/best.pt'
model2_path = './license_plate_noisy_v5/yolov8_finetuned_detection_noisy_v5/weights/best.pt'

data_path = './datasets/detection_noisy_v5/test/images'

def main():
    
    model1 = YOLO(model1_path)
    model2 = YOLO(model2_path)

    # 모델 1 결과 저장
    model1.predict(source=data_path, save=True, project='comparison', name='model1_results')

    # 모델 2 결과 저장
    model2.predict(source=data_path, save=True, project='comparison', name='model2_results')

    model1.val(data='./detection.yaml')
    model2.val(data='./detection.yaml')

if __name__ == '__main__':
    main()