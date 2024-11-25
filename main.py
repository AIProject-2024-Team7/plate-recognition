from ultralytics import YOLO
import yaml


def main():
    model = YOLO('yolov8n.pt')  

    model.train(
        data='./recognition_data.yaml',  
        epochs=200,                      
        batch=16,                        
        imgsz=640,                       
        optimizer='Adam',                
        project='팀플',   
        name='팀플',         
        device=0                         
    )

    # 모델 검증
    metrics = model.val()
    print(metrics)  

    results = model('./data/train/images')  
    results.show()  

if __name__ == '__main__':
    main()


