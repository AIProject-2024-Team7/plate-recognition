from ultralytics import YOLO
import yaml


def main():
    model = YOLO('yolov8n.pt') 

    model.train(
        data='./detection.yaml',        
        epochs=500,               
        batch=16,                
        imgsz=640,               
        optimizer='Adam',        
        project='license_plate', 
        name='yolov8_finetuned_detection',  
        device=0                 
    )

    metrics = model.val()
    print(metrics)

if __name__ == '__main__':
    main()